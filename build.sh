#!/bin/bash

cd dnn
chmod +rwx ./build.sh
./build.sh
cd ..
cd Convnet
dotnet build Convnet.csproj -c:Release
