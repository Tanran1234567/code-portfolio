#!/usr/bin/env python
# -*-coding: utf-8-*-

import os

import dramatiq
import dramatiq.results.backends as rst
import dramatiq_abort.backends as abt
from dramatiq.brokers.redis import RedisBroker
from dramatiq.message import Message, global_encoder
from dramatiq.middleware import AgeLimit, AsyncIO, Callbacks, CurrentMessage
from dramatiq.results import Results
from dramatiq_abort import Abortable

from mind_zone.utils import Logger


def init_broker():
    namespace = "dramatiq-{}-{}".format(os.getenv("MZ_SVC_NAME").replace("-", "_"), os.getenv("MZ_ENV", "dev"))
    broker = RedisBroker(
        namespace=namespace,
        host=os.getenv("MZ_BROKER_HOST"),
        port=os.getenv("MZ_BROKER_PORT"),
        db=os.getenv("MZ_BROKER_DB"),
        password=os.getenv("MZ_BROKER_PASSWD"),
    )

    broker.add_middleware(AsyncIO(), before=AgeLimit)
    broker.add_middleware(
        Results(backend=rst.RedisBackend(client=broker.client, namespace=f"{namespace}:results"), result_ttl=3600000),
        before=Callbacks,
    )
    broker.add_middleware(
        Abortable(backend=abt.RedisBackend(client=broker.client, namespace=f"{namespace}:aborts:")),
        before=Callbacks,
    )
    broker.add_middleware(CurrentMessage(), before=Callbacks)

    dramatiq.set_broker(broker)
    Logger.info("Use {} ({}) as dramatiq broker for background tasks".format(type(broker).__name__, hex(id(broker))))
    return broker


Broker = init_broker()


# NOTE: deprecated in future version
def encode_message(message: Message, excluded: list[str] = ["args", "kwargs", "options"]) -> bytes:
    msg_dict = message.asdict()
    for field in excluded:
        match field:
            case "args":
                msg_dict[field] = ()
            case "kwargs" | "options":
                msg_dict[field] = {}
            case _:
                msg_dict[field] = None

    return global_encoder.encode(msg_dict)


# NOTE: deprecated in future version
def get_task_result(encoded_msg: bytes, *, block=False, timeout=None):
    global Broker
    message = Message.decode(encoded_msg)
    return Broker.get_results_backend().get_result(message, block=block, timeout=timeout)
