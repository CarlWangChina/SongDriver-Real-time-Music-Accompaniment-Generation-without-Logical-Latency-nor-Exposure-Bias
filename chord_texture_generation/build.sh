em++ -I./lua main.cpp ./lua/src/*.o ./lua-cjson/*.o -std=c++20 -O3\
 -o chordScript.js --embed-file scripts
cp chordScript.js ../frontend/public
cp chordScript.wasm ../frontend/public