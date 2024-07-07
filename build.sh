#!/bin/bash

cd dnn
chmod +rwx ./build.sh
./build.sh
cd ..
cd ConvnetAvalonia
dotnet build ConvnetAvalonia.csproj -c:Release
