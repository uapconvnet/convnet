cd dnn
call build.cmd
cd ..
dotnet restore
msbuild Convnet.sln /p:Configuration=Debug
msbuild Convnet.sln /p:Configuration=Release