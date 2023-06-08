python3 conv1.py test/testconv1.txt > test/testconv1.out.txt #转换模型输出的格式
./chordPlayer test/testconv1.out.txt test/testconv1.play.mid #演奏

python3 conv2.py test/testconv2.txt  test/testconv2.out.txt #转换模型输出的格式
./chordPlayer test/testconv2.out.txt test/testconv2.play.mid #演奏
