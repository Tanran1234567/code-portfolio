#!/bin/bash

ROOT=$(cd `dirname $0`; pwd)
BIN_DIR="${ROOT}/.venv/bin"

mkdir -p ${ROOT}/logs

get_run_mode() {
    echo -e "请选择服务运行模式: \n    1) dev\n    2) test\n    3) prod"
    read -p "请输入 1, 2, 3, dev, test, prod 之一: " run_mode
    echo ""
}

MZ_ENV=$1

while [[ -z "${MZ_ENV}" ]] || ([[ ${MZ_ENV} != "dev" ]] && [[ ${MZ_ENV} != "test" ]] && [[ ${MZ_ENV} != "prod" ]]); do
    get_run_mode
    case ${run_mode} in
        1|dev)
            MZ_ENV="dev"
            ;;
        2|test)
            MZ_ENV="test"
            ;;
        3|prod)
            MZ_ENV="prod"
            ;;
        *)
            echo "无效的选项，请输入 1, 2, 3, dev, test, prod 之一"
            ;;
    esac
done

SVC_NAME='iassistant'
echo "正在以 ${MZ_ENV} 模式启动 ${SVC_NAME} 服务..."
if [[ ! -f "${ROOT}/config/.env.${MZ_ENV}" ]]; then
    echo "请提供环境配置文件'${ROOT}/config/.env.${MZ_ENV}'"
    exit 1
fi

CMD="cd ${ROOT}/src"
if [[ ${MZ_ENV} == 'dev' ]]; then
    CMD="${CMD} && MZ_ENV=${MZ_ENV} ${BIN_DIR}/dramatiq --processes 2 worker"
    nohup bash -c "${CMD} 1>../logs/dramatiq.log 2>&1" >/dev/null 2>&1 &
else
    CMD="${CMD} && MZ_ENV=${MZ_ENV} ${BIN_DIR}/dramatiq worker"
    nohup bash -c "${CMD} 1>>../logs/dramatiq.log 2>&1" >/dev/null 2>&1 &
fi

CMD="cd ${ROOT}/src && MZ_ENV=${MZ_ENV} ${BIN_DIR}/python -m main"
if [[ ${MZ_ENV} == 'dev' ]]; then
    nohup bash -c "${CMD} 1>../logs/fastapi.log 2>&1" >/dev/null 2>&1 &
else
    bash -c "${CMD} 1>>../logs/fastapi.log 2>&1" >/dev/null 2>&1
fi
