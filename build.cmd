cd dnn
call build.cmd
cd ..
msbuild Convnet.sln /p:Configuration=Debug
msbuild Convnet.sln /p:Configuration=Release