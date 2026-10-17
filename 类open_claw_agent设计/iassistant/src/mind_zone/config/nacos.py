#!/usr/bin/env python
# -*-coding: utf-8-*-

import os
import random
from pathlib import Path
from typing import Union

import yaml
from v2.nacos import ClientConfigBuilder, DeregisterInstanceParam, GRPCConfig, NacosNamingService, RegisterInstanceParam

from mind_zone.utils import Logger, get_host_addr


async def create_nacos_client(conf_file: Union[str, Path], level: str = "INFO"):
    with open(conf_file, "r", encoding="utf-8") as f:
        conf_data = yaml.safe_load(f.read())

    app_home = Path(conf_file).parent.parent
    nacos_config = conf_data.get("nacos", {}).get(os.getenv("MZ_ENV", "dev"), {})
    client_config = (
        ClientConfigBuilder()
        .server_address(f"{nacos_config['host']}:{nacos_config['port']}")
        .namespace_id(nacos_config["namespace"])
        .username(nacos_config["username"])
        .password(os.getenv("MZ_NACOS_PASSWD"))
        .cache_dir(str(app_home / "nacos" / "cache"))
        .log_dir(str(app_home / "nacos" / "logs"))
        .log_level(level.upper())
        .grpc_config(GRPCConfig(grpc_timeout=5000))
        .build()
    )

    try:
        return await NacosNamingService.create_naming_service(client_config)
    except Exception as exc:
        Logger.error(f"Failed in creating NACOS naming service: {exc}")
        if os.getenv("MZ_ENV", "dev") != "dev":
            raise exc


async def register_service(client: NacosNamingService, ephemeral: bool = True):
    service_name = os.getenv("MZ_SVC_NAME", "mind-zone")
    try:
        if os.getenv("MZ_ENV", "dev") != "dev" and os.getenv("MZ_NACOS_HOSTS"):
            host = random.choice(os.getenv("MZ_NACOS_HOSTS").split(","))
        else:
            host = get_host_addr()
        port = int(os.getenv("MZ_NACOS_PORT", 8080))
        response = await client.register_instance(
            request=RegisterInstanceParam(
                service_name=service_name,
                ip=host,
                port=port,
                weight=1.0,
                enabled=True,
                healthy=True,
                ephemeral=ephemeral,
            )
        )
        if not response:
            raise RuntimeError("NO response from registering instance")
        Logger.info(f"Register {service_name} [{host}:{port}] to NACOS")
        return host, port
    except Exception as exc:
        Logger.error(f"Failed in registering service to NACOS: {exc}")
        if os.getenv("MZ_ENV", "dev") != "dev":
            raise exc
        return "", 0


async def deregister_service(client: NacosNamingService, host: str, port: int, ephemeral: bool = True):
    service_name = os.getenv("MZ_SVC_NAME", "mind-zone")
    try:
        response = await client.deregister_instance(
            request=DeregisterInstanceParam(service_name=service_name, ip=host, port=port, ephemeral=ephemeral)
        )
        if not response:
            raise RuntimeError("NO response from deregistering instance")
        Logger.info(f"Deregister {service_name} [{host}:{port}] from NACOS")
    except Exception as exc:
        Logger.error(f"Failed in deregistering service from NACOS: {exc}")
        if os.getenv("MZ_ENV", "dev") != "dev":
            raise exc
