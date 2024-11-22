ECHO OFF
SET mode=%1
IF mode EQU "" SET mode="Release"
mkdir build
cd build
cmake -A x64 .. -DCMAKE_BUILD_TYPE=%mode% -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
msbuild dnn.sln /p:Configuration=%mode%
cd ..