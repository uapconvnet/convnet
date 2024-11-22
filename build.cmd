cd dnn
call build.cmd %1
cd ..
dotnet restore
msbuild Convnet.sln /p:Configuration=%1
