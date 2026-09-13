#!/usr/bin/env python3
from websockify.websocketproxy import ProxyRequestHandler, WebSocketProxy


class NoCacheProxyRequestHandler(ProxyRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    WebSocketProxy(
        RequestHandlerClass=NoCacheProxyRequestHandler,
        listen_host="0.0.0.0",
        listen_port=6080,
        target_host="localhost",
        target_port=5900,
        web="/usr/share/novnc/",
    ).start_server()
