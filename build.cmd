ECHO OFF
SET mode=%1
IF mode EQU "" SET mode="Release"
cd dnn
call build.cmd %1
cd ..
dotnet restore
msbuild Convnet.sln /p:Configuration=%mode%