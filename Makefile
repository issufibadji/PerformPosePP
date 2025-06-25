TARGET=openpose_image_coco
CXX=g++
CXXFLAGS=`pkg-config --cflags --libs opencv4`

$(TARGET): OpenPoseImageCoco.cpp
	$(CXX) OpenPoseImageCoco.cpp -o $(TARGET) $(CXXFLAGS)

clean:
	rm -f $(TARGET)
