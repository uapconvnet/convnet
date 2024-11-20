#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd dnn
chmod +rwx ./build.sh
./build.sh
cd ..
cd Convnet
dotnet restore
dotnet build Convnet.csproj -c:Release
cd ..

echo "[Desktop Entry]
Encoding=UTF-8
Version=1.0
Type=Application
Terminal=false
Exec=${SCRIPT_DIR}/Convnet/bin/Release/net9.0/Convnet
Name=Convnet
Icon=${SCRIPT_DIR}/Convnet/Resources/App.ico
Comment=
NoDisplay=false" > ${HOME}/.local/share/applications/convnet.desktop
