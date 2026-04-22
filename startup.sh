#!/bin/bash

# ==============================
# script : startup.sh
# desc   : docker-compose 실행
# create : 2026-04-22
# modify : --
# author : Seonghak Kim(ksh0306@gmail.com)
# ==============================
export CURRENT_UID=$(id -u)
export CURRENT_GID=$(id -g)

echo "Starting container with UID: $CURRENT_UID, GID: $CURRENT_GID"

docker compose up -d --build

echo "Container is running!"