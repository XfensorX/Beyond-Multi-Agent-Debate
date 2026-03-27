"""
Implements a simple Round Robin Reverse Proxy, as Nginx on the cluster is neither available
nor runnable through apptainer, as no-root user cannot use the host network.

"""

import argparse
import logging
import socket
import threading
from itertools import cycle

import httpx
from fastapi import FastAPI, Request

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


def start_http_proxy(http_ports, listen_port):
    app = FastAPI()
    instance_cycle = cycle(http_ports)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy(request: Request, path: str):
        target = next(instance_cycle)
        url = f"http://127.0.0.1:{target}/{path}"
        logging.info(f"Forwarding HTTP {request.method} {path} -> {url}")
        async with httpx.AsyncClient() as client:
            resp = await client.request(
                request.method,
                url,
                headers=request.headers.raw,
                content=await request.body(),
                timeout=None,
            )
        return resp.content, resp.status_code, resp.headers

    import uvicorn

    logging.info(
        f"Starting HTTP proxy on 127.0.0.1:{listen_port} forwarding {http_ports}"
    )
    uvicorn.run(app, host="127.0.0.1", port=listen_port, log_level="info")


class TCPProxyServer:
    def __init__(self, listen_port, target_ports):
        self.listen_port = listen_port
        self.target_ports = target_ports
        self.target_cycle = cycle(target_ports)

    def start(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", self.listen_port))
        server.listen(100)
        logging.info(
            f"Starting TCP/gRPC proxy on 127.0.0.1:{self.listen_port} forwarding {self.target_ports}"
        )
        while True:
            client_sock, addr = server.accept()
            target_port = next(self.target_cycle)
            target_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                target_sock.connect(("127.0.0.1", target_port))
                threading.Thread(
                    target=self.forward, args=(client_sock, target_sock), daemon=True
                ).start()
                threading.Thread(
                    target=self.forward, args=(target_sock, client_sock), daemon=True
                ).start()
            except Exception as e:
                logging.error(f"Connection error: {e}")
                client_sock.close()
                target_sock.close()

    @staticmethod
    def forward(source, dest):
        try:
            while True:
                data = source.recv(4096)
                if not data:
                    break
                dest.sendall(data)
        except Exception:
            pass
        finally:
            source.close()
            dest.close()


# ----------- CLI -----------


def main():
    parser = argparse.ArgumentParser(description="Phoenix user-space HTTP + gRPC proxy")
    parser.add_argument(
        "--http-ports",
        nargs="+",
        type=int,
        required=True,
        help="List of Phoenix HTTP ports",
    )
    parser.add_argument("--http-listen", type=int, help="Local HTTP proxy listen port")
    parser.add_argument(
        "--grpc-ports",
        nargs="+",
        type=int,
        required=True,
        help="List of Phoenix gRPC ports",
    )
    parser.add_argument("--grpc-listen", type=int, help="Local gRPC proxy listen port")
    args = parser.parse_args()

    http_thread = threading.Thread(
        target=start_http_proxy, args=(args.http_ports, args.http_listen), daemon=True
    )
    http_thread.start()

    grpc_proxy = TCPProxyServer(
        listen_port=args.grpc_listen, target_ports=args.grpc_ports
    )
    grpc_proxy.start()


if __name__ == "__main__":
    main()
