@ECHO OFF

IF "%~1"=="" GOTO release

cd dnn
call build.cmd %1
cd ..
dotnet restore
msbuild Convnet.sln /p:Configuration=%1
GOTO :EOF

:release
cd dnn
call build.cmd "Release"
cd ..
dotnet restore
msbuild Convnet.sln /p:Configuration=Release
