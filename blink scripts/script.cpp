kernel Custom_ColorMult : ImageComputationKernel<ePixelWise>
{
  Image<eRead,eAccessRandom,eEdgeClamped> src;                       //input image with edges clamped 
  Image<eWrite> dst;                                                //output image
  
  param:
    float4 color; 
    int radius;                                                  //parameter
  
  void define() {
    defineParam(color, "Custom color", float4(0.0f,1.0f,0.0f,0.0f));//default value
    defineParam(radius, "radius", 3);
  }
  

  void init() {
    // src.setRange(2, 3); // Set a range of x-2, x+2 and y-3, y+3.
  }


  void process(int2 pos) {
    //int radius =3;
    //image.SampleType val = src(int3(pos.x-1,pos.y,0));
    //float4 = src(pos.x,pos.y);
    float4 total = float4(0.0f, 0.0f, 0.0f, 0.0f); // Initialize to zero
    int n = 0;
    total = total + src(pos.x - 1, pos.y - 1);
    for(int i= -radius; i <= radius; i++) {
     for(int j = -radius; j <= radius; j++) {
        total = total + src(pos.x + i, pos.y + j);
        n++;
     }    
    }

    float4 srcPixel = src(pos.x, pos.y);
    float4 srcPixel2 = src(pos.x - 1, pos.y - 1);
    float4 total2 = srcPixel + srcPixel2;
    dst() = total / n;
  }
};