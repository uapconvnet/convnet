mkdir build
cd build
cmake -A x64 .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
msbuild dnn.sln /p:Configuration=Debug
msbuild dnn.sln /p:Configuration=Release
cd ..