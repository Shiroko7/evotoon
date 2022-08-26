#include <math.h>

/* period parameters */
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL   /*!< constant vector a */
#define UPPER_MASK 0x80000000UL /*!< most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL /*!< least significant r bits */

static unsigned long mt[N]; /* the array for the state vector  */
static int mti=N+1; /* mti==N+1 means mt[N] is not initialized */

char whichGaussian=0; 

/*===============================================================*/
/**
   Initializes mt[N] with a seed s.
*/
/*===============================================================*/

void setSeed(unsigned long s)
{
    mt[0]= s & 0xffffffffUL;
    for (mti=1; mti<N; mti++) {
        mt[mti] = 
	    (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti); 
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
        mt[mti] &= 0xffffffffUL;
        /* for >32 bit machines */
    }
}

/*===============================================================*/
/**
   Generates a random number on [0,0xffffffff]-interval
*/
/*===============================================================*/

unsigned long genrand_int32(void)
{
    unsigned long y;
    static unsigned long mag01[2]={0x0UL, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti >= N) { /* generate N words at one time */
        int kk;
 
        if (mti == N+1)   /* if init_genrand() has not been called, */
            setSeed(5489UL); /* a default initial seed is used */

        for (kk=0;kk<N-M;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (;kk<N-1;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
        mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

        mti = 0;
    }
  
    y = mt[mti++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}

/*===============================================================*/
/**
   Generates a random number on [0,1)-real-interval
*/
/*===============================================================*/

double drand()
{
    return genrand_int32()*(1.0/4294967296.0); 
    /* divided by 2^32 */
}

/*===============================================================*/
/**
   Returns random integer from 0 to max-1.
*/
/*===============================================================*/

int intRand(int max)
{
  return (int) ((double)drand()*max);
};

double gaussianRandom(double mean,double stddev)
{
   double  q,u,v,x,y;

   /*
      Generate P = (u,v) uniform in rect. enclosing acceptance region
      Make sure that any random numbers <= 0 are rejected, since
      gaussian() requires uniforms > 0, but RandomUniform() delivers >= 0.
   */
   do {
      do { u=drand(); } while (u==0);
      do { v=drand(); } while (v==0);

      v = 1.7156 * (v - 0.5);

      /*  Evaluate the quadratic form */
      x = u - 0.449871;
      y = fabs(v) + 0.386595;
      q = x * x + y * (0.19600 * y - 0.25472 * x);

      /* Accept P if inside inner ellipse */
      if (q < 0.27597)
         break;

      /*  Reject P if outside outer ellipse, or outside acceptance region */
    } while ((q > 0.27846) || (v * v > -4.0 * log(u) * u * u));

    /*  Return ratio of P's coordinates as the normal deviate */
    return (mean + stddev * v / u);
}

