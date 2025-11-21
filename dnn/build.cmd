@ECHO OFF

rmdir /s /q .\build

IF "%~1"=="" GOTO release

mkdir build
cd build
cmake -G "Visual Studio 18 2026" -A x64 .. -DCMAKE_BUILD_TYPE=%1 -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.10
msbuild dnn.slnx /p:Configuration=%1
cd ..
GOTO :EOF

:release
mkdir build
cd build
cmake -G "Visual Studio 18 2026" -A x64 .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.10
msbuild dnn.slnx /p:Configuration=Release
cd ..


