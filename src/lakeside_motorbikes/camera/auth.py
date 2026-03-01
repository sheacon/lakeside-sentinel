import logging

import glocaltokens.client

logger = logging.getLogger(__name__)

NEST_SCOPE = "oauth2:https://www.googleapis.com/auth/nest-account"


class _MultiServiceAuth(glocaltokens.client.GLocalAuthenticationTokens):
    """Extends glocaltokens to support fetching tokens for arbitrary OAuth scopes."""

    _last_service: str | None = None

    def get_access_token(
        self, service: str = glocaltokens.client.ACCESS_TOKEN_SERVICE
    ) -> str | None:
        if (
            self.access_token is not None
            and self._last_service == service
            and not self.access_token_expired
        ):
            return self.access_token

        master_token = self.get_master_token()
        if master_token is None:
            logger.error("Failed to obtain master token")
            return None

        res = glocaltokens.client.perform_oauth(
            self._escape_username(self.username),
            master_token,
            self.get_android_id(),
            app=glocaltokens.client.ACCESS_TOKEN_APP_NAME,
            service=service,
            client_sig=glocaltokens.client.ACCESS_TOKEN_CLIENT_SIGNATURE,
        )

        if "Auth" not in res:
            logger.error("OAuth response missing 'Auth' key: %s", res)
            return None

        self.access_token = res["Auth"]
        self.access_token_date = glocaltokens.client.datetime.now()
        self._last_service = service
        return self.access_token


class NestAuth:
    """Manages authentication for the Google Nest internal API."""

    def __init__(self, master_token: str, username: str) -> None:
        self._auth = _MultiServiceAuth(
            master_token=master_token,
            username=username,
            password="UNUSED",
        )

    def get_access_token(self) -> str:
        """Get a valid Nest-scoped access token, refreshing if needed."""
        token = self._auth.get_access_token(service=NEST_SCOPE)
        if token is None:
            raise RuntimeError("Failed to obtain Nest access token")
        return token
