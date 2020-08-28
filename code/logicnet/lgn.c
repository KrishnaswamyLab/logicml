/**CFile****************************************************************

  FileName    [lgnYYMMDD.c]

  SystemName  [ABC: Logic synthesis and verification system.]

  PackageName [LogicNet implementation.]

  Synopsis    [Compiling "gcc -DLIN64 -o lgn lgnYYMMDD.c"]

  Author      [Alan Mishchenko]
  
  Affiliation [UC Berkeley]

  Date        [Ver. 1.0. Started - November 2019.]

  Revision    [$Id: lgnYYMMDD.c,v 1.00 2005/06/20 00:00:00 alanmi Exp $]

***********************************************************************/

/*
    ABC: System for Sequential Synthesis and Verification

    http://www.eecs.berkeley.edu/~alanmi/abc/


    Copyright (c) The Regents of the University of California. All rights reserved.

    Permission is hereby granted, without written agreement and without license or
    royalty fees, to use, copy, modify, and distribute this software and its
    documentation for any purpose, provided that the above copyright notice and
    the following two paragraphs appear in all copies of this software.

    IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR
    DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF
    THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF
    CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
    BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS,
    AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATION TO PROVIDE MAINTENANCE,
    SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "abc_global.h"
#include "utilTruth.h"
#include "vecBit.h"
#include "vecInt.h"
#include "vecPtr.h"
#include "vecStr.h"
#include "vecWec.h"
#include "vecWrd.h"

ABC_NAMESPACE_IMPL_START

////////////////////////////////////////////////////////////////////////
///                        DECLARATIONS                              ///
////////////////////////////////////////////////////////////////////////

#define LGN_VER_NUM   200116 
#define LGN_LINE_MAX    1000
 
// representation of a data set composed of nSamples of data,
// each sample contains nSampleBits bits
typedef struct Lgn_Data_t_ Lgn_Data_t;
struct Lgn_Data_t_
{
    int             nDims;        // data dimensions
    int             nSamples;     // number of data samples
    int             nSampleBits;  // number of bits in each sample
    int             nDataWords;   // nDataWords  = Abc_Bit6WordNum(nSamples*nSampleBits)
    word *          pData;        // beginning of sample data   
};

// representation of LogicNet
typedef struct Lgn_Net_t_ Lgn_Net_t;
struct Lgn_Net_t_
{
    Lgn_Data_t      DataTX;       // training input data matrix  (nSamples x nBitsIn  where nBitsIn  can be SizeVert*SizeHor)
    Lgn_Data_t      DataTY;       // training output data matrix (nSamples x nBitsOut where nBitsOut can be 1 for binary MNIST)
    Lgn_Data_t      DataVX;       // validation input data matrix
    Lgn_Data_t      DataVY;       // validation output data matrix
    Vec_Ptr_t       vLayers;      // array of pointers to Lgn_Layer_t
    Vec_Wrd_t       vSimsPi;      // array of output simulation info
    Vec_Str_t *     vOutValue;    // array of output values for each pattern
    word **         pOutValue;    // bit-vector of output values for each pattern
    int             nSimWords;    // nSimWords = Abc_Bit6WordNum(pData->nSamples)
};
 
// representation of one layer
typedef struct Lgn_Layer_t_ Lgn_Layer_t;
struct Lgn_Layer_t_
{
    int             Id;           // layer number
    int             nLuts;        // LUT count 
    int             LutSize;      // LUT size 
    int             nTruthWords;  // nTruthWords = Abc_Bit6WordNum(1 << LutSize)
    Vec_Int_t       vMark;        // array has nLuts 32-bit marks
    Vec_Int_t       vFanins;      // array has nLuts*LutSize 32-bit integers
    Vec_Int_t       vPolars;      // array has nLuts polarities (0=neg, 1=pos, 2=unused)
    Vec_Wec_t       vFanouts;     // array has nLuts arrays with fanouts of each LUT
    Vec_Wec_t       vTfoNodes;    // array has nLuts arrays with fanouts of each LUT
    Vec_Wrd_t       vTruths;      // array has nLuts*nTruthWords 64-bit truth table words
    Vec_Wrd_t       vSims;        // array has nLuts*pNet->nSimWords 64-bit simulation words
    Vec_Wrd_t       vCare;        // array has nLuts*pNet->nSimWords 64-bit simulation words
    Lgn_Net_t *     pNet;         // parent network
};

static inline int           Lgn_DataBit       ( Lgn_Data_t * p, int s, int b ) { return Abc_TtGetBit(p->pData, p->nSampleBits*s + b);                  }
 
static inline word *        Lgn_LayerLutTruth ( Lgn_Layer_t * p, int i )       { return Vec_WrdEntryP(&p->vTruths, p->nTruthWords*i);                  }
static inline word *        Lgn_LayerLutSims  ( Lgn_Layer_t * p, int i )       { return Vec_WrdEntryP(&p->vSims,   p->pNet->nSimWords*i);              }
static inline word *        Lgn_LayerLutCare  ( Lgn_Layer_t * p, int i )       { return Vec_WrdEntryP(&p->vCare,   p->pNet->nSimWords*i);              }
static inline int           Lgn_LayerLutPolar ( Lgn_Layer_t * p, int i )       { return Vec_IntEntry(&p->vPolars,  i);                                 }
static inline Lgn_Layer_t * Lgn_LayerPrev     ( Lgn_Layer_t * p )              { return p->Id ? (Lgn_Layer_t *)Vec_PtrEntry(&p->pNet->vLayers, p->Id-1) : NULL;                              }
static inline Lgn_Layer_t * Lgn_LayerNext     ( Lgn_Layer_t * p )              { return p->Id < Vec_PtrSize(&p->pNet->vLayers)-1 ? (Lgn_Layer_t *)Vec_PtrEntry(&p->pNet->vLayers, p->Id+1) : NULL; }
static inline int           Lgn_LayerIsLast   ( Lgn_Layer_t * p )              { return p->Id == Vec_PtrSize(&p->pNet->vLayers) - 1;                   } 

static inline word *        Lgn_NetSimsIn     ( Lgn_Net_t * p, int i )         { return Vec_WrdEntryP(&p->vSimsPi, p->nSimWords*i);                    }
static inline int           Lgn_NetLayerNum   ( Lgn_Net_t * p )                { return Vec_PtrSize(&p->vLayers);                                      }
static inline Lgn_Layer_t * Lgn_NetLayer      ( Lgn_Net_t * p, int i )         { return i < Vec_PtrSize(&p->vLayers) ? (Lgn_Layer_t *)Vec_PtrEntry(&p->vLayers, i) : NULL;   }
static inline Lgn_Layer_t * Lgn_NetLayerLast  ( Lgn_Net_t * p )                { return (Lgn_Layer_t *)Vec_PtrEntry(&p->vLayers, Lgn_NetLayerNum(p)-1);}
static inline int           Lgn_NetLayerIsLast( Lgn_Net_t * p, int i )         { return i == Vec_PtrSize(&p->vLayers) - 1;                             }

#define Lgn_NetForEachLayer( p, pLayer, i )            \
    Vec_PtrForEachEntry( Lgn_Layer_t *, &p->vLayers, pLayer, i )
#define Lgn_NetForEachLayerReverse( p, pLayer, i )     \
    Vec_PtrForEachEntryReverse( Lgn_Layer_t *, &p->vLayers, pLayer, i )

#define Lgn_LayerForEachLut( pLayer, iLut )            \
    for ( iLut = 0; iLut < pLayer->nLuts; iLut++ )

#define Lgn_LutForEachFanin( pLayer, iLut, iFanin, k ) \
    for ( k = 0; k < pLayer->LutSize && (((iFanin) = Vec_IntEntry(&pLayer->vFanins, iLut*pLayer->LutSize+k)), 1); k++ )

#define Lgn_LutForEachFanout( pLayer, iLut, iFanout, k ) \
    for ( k = 0; k < Vec_IntSize(Vec_WecEntry(&pLayer->vFanouts, iLut)) && (((iFanout) = Vec_IntEntry(Vec_WecEntry(&pLayer->vFanouts, iLut), k)), 1); k++ )

#define Lgn_LutForEachTfoNode( pLayer, iLut, iNode, k ) \
    for ( k = 0; k < Vec_IntSize(Vec_WecEntry(&pLayer->vTfoNodes, iLut)) && (((iNode) = Vec_IntEntry(Vec_WecEntry(&pLayer->vTfoNodes, iLut), k)), 1); k++ )

////////////////////////////////////////////////////////////////////////
///                     FUNCTION DEFINITIONS                         ///
////////////////////////////////////////////////////////////////////////


/**Function*************************************************************

  Synopsis    [Converts MNIST to 1-bit pixels and binary output {0-4} vs {5-9}.]

  Description [http://yann.lecun.com/exdb/mnist/]
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
void Lgn_NetMnistConvert() 
{
    char * pFileNameInTX  = "data/mnist/train-images.idx3-ubyte";
    char * pFileNameInTY  = "data/mnist/train-labels.idx1-ubyte";
    char * pFileNameInVX  = "data/mnist/t10k-images.idx3-ubyte";
    char * pFileNameInVY  = "data/mnist/t10k-labels.idx1-ubyte";

    char * pFileNameOutTX = "data/mnist/mnist_1k_28_28_1_1.data";
    char * pFileNameOutTY = "data/mnist/mnist_1k_1.data";
    char * pFileNameOutVX = "data/mnist/mnist_10k_28_28_1_1.data";
    char * pFileNameOutVY = "data/mnist/mnist_10k_1.data";

    int n, nSize[2] = {1000, 10000};
    for ( n = 0; n < 2; n++ )
    {
        FILE * pFileOutX  = fopen( n ? pFileNameOutVX : pFileNameOutTX, "wb" );
        FILE * pFileOutY  = fopen( n ? pFileNameOutVY : pFileNameOutTY, "wb" );

        char * pMemoryX   = Abc_FileReadContents( n ? pFileNameInVX : pFileNameInTX, NULL );
        char * pMemoryY   = Abc_FileReadContents( n ? pFileNameInVY : pFileNameInTY, NULL );

        Vec_Bit_t * vBitX = Vec_BitAlloc( nSize[n] * 28 * 28 * 1 * 1 );
        Vec_Bit_t * vBitY = Vec_BitAlloc( nSize[n] * 1 );

        int k, x, y, Num, Value, nBytes;
        for ( k = 0; k < nSize[n]; k++ )
        {
            for ( y = 0; y < 28; y++ )
            for ( x = 0; x < 28; x++ )
                Vec_BitPush( vBitX, (pMemoryX[16+k*28*28+y*28+x] >> 7) & 1 );  // binary data
            Vec_BitPush( vBitY, (pMemoryY[8+k] >= 5) & 1 );                    // binary labels
            //for ( y = 0; y < 10; y++ )
            //    Vec_BitPush( vBitY, (int)(y == pMemoryY[8+k]) );               // 10-valued labels
        }
        assert( Vec_BitSize(vBitX) == Vec_BitCap(vBitX) );
        assert( Vec_BitSize(vBitY) <= Vec_BitCap(vBitY) );

        Num = 5;         Value = fwrite( &Num, 1, 4, pFileOutX ); assert( Value == 4 );
        Num = nSize[n];  Value = fwrite( &Num, 1, 4, pFileOutX ); assert( Value == 4 );
        Num = 28;        Value = fwrite( &Num, 1, 4, pFileOutX ); assert( Value == 4 );
        Num = 28;        Value = fwrite( &Num, 1, 4, pFileOutX ); assert( Value == 4 );
        Num = 1;         Value = fwrite( &Num, 1, 4, pFileOutX ); assert( Value == 4 );
        Num = 1;         Value = fwrite( &Num, 1, 4, pFileOutX ); assert( Value == 4 );

        nBytes = nSize[n] * 28 * 28 * 1 * 1 / 8;
        assert( nSize[n] * 28 * 28 * 1 * 1 % 8 == 0 );
        Value = fwrite( Vec_BitArray(vBitX), 1, nBytes, pFileOutX );
        assert( Value == nBytes );


        Num = 2;         Value = fwrite( &Num, 1, 4, pFileOutY ); assert( Value == 4 );
        Num = nSize[n];  Value = fwrite( &Num, 1, 4, pFileOutY ); assert( Value == 4 );
        Num = 1;         Value = fwrite( &Num, 1, 4, pFileOutY ); assert( Value == 4 );

        nBytes = nSize[n] * 1 / 8;
        assert( nSize[n] * 1 % 8 == 0 );
        Value = fwrite( Vec_BitArray(vBitY), 1, nBytes, pFileOutY );
        assert( Value == nBytes );

        Vec_BitFree( vBitX );
        Vec_BitFree( vBitY );

        fclose( pFileOutX );
        fclose( pFileOutY );
    }
}


/**Function*************************************************************

  Synopsis    [Converts CIFAR-10 to 1-bit pixels and binary output {0-4} vs {5-9}.]

  Description [https://www.cs.toronto.edu/~kriz/cifar.html]
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
void Lgn_NetCifar10Convert()
{
    // Each file contains 10000 such 3073-byte "rows" of images, 
    // although there is nothing delimiting the rows. Therefore 
    // each file should be exactly 30730000 bytes long. 
    char * pFileName     =   "data/cifar10/test_batch.bin";
    char * pFileNames[5] = { "data/cifar10/data_batch_1.bin",
                             "data/cifar10/data_batch_2.bin",
                             "data/cifar10/data_batch_3.bin",
                             "data/cifar10/data_batch_4.bin",
                             "data/cifar10/data_batch_5.bin" };

    char * pFileNameOutTX = "data/cifar10/cifar10_50k_32_32_3_1.data";
    char * pFileNameOutTY = "data/cifar10/cifar10_50k_1.data";
    char * pFileNameOutVX = "data/cifar10/cifar10_10k_32_32_3_1.data";
    char * pFileNameOutVY = "data/cifar10/cifar10_10k_1.data";
    int n, i, k, x, y, Num, Value, nBytes;

    int nSize[2] = {50000, 10000};
    for ( n = 0; n < 2; n++ )
    {
        FILE * pFileOutX = fopen( n ? pFileNameOutVX : pFileNameOutTX, "wb" );
        FILE * pFileOutY = fopen( n ? pFileNameOutVY : pFileNameOutTY, "wb" );

        Vec_Bit_t * vBitX = Vec_BitAlloc( nSize[n] * 32 * 32 * 3 * 1 );
        Vec_Bit_t * vBitY = Vec_BitAlloc( nSize[n] * 1 );
        for ( i = 0; i < nSize[n]/10000; i++ )
        {
            FILE * pFile = fopen( n ? pFileName : pFileNames[i], "rb" );
            for ( k = 0; k < 10000; k++ )
            {
                unsigned char pBuffer[3100];
                Value = fread( pBuffer, 1, 3073, pFile );
                assert( Value == 3073 );
                Vec_BitPush( vBitY, (pBuffer[0] >= 5) & 1 );
                for ( y = 0; y < 32; y++ )
                for ( x = 0; x < 32; x++ )
                {
                    Vec_BitPush( vBitX, (pBuffer[1+1024*0+y*32+x] >> 7) & 1 );
                    Vec_BitPush( vBitX, (pBuffer[1+1024*1+y*32+x] >> 7) & 1 );
                    Vec_BitPush( vBitX, (pBuffer[1+1024*2+y*32+x] >> 7) & 1 );
                }
            }
            fclose( pFile );
        }
        assert( Vec_BitSize(vBitX) == Vec_BitCap(vBitX) );
        assert( Vec_BitSize(vBitY) <= Vec_BitCap(vBitY) );

        Num = 5;         Value = fwrite( &Num, 1, 4, pFileOutX ); assert( Value == 4 );
        Num = nSize[n];  Value = fwrite( &Num, 1, 4, pFileOutX ); assert( Value == 4 );
        Num = 32;        Value = fwrite( &Num, 1, 4, pFileOutX ); assert( Value == 4 );
        Num = 32;        Value = fwrite( &Num, 1, 4, pFileOutX ); assert( Value == 4 );
        Num = 3;         Value = fwrite( &Num, 1, 4, pFileOutX ); assert( Value == 4 );
        Num = 1;         Value = fwrite( &Num, 1, 4, pFileOutX ); assert( Value == 4 );

        nBytes = nSize[n] * 32 * 32 * 3 * 1 / 8;
        assert( nSize[n] * 32 * 32 * 3 * 1 % 8 == 0 );
        Value = fwrite( Vec_BitArray(vBitX), 1, nBytes, pFileOutX );
        assert( Value == nBytes );


        Num = 2;         Value = fwrite( &Num, 1, 4, pFileOutY ); assert( Value == 4 );
        Num = nSize[n];  Value = fwrite( &Num, 1, 4, pFileOutY ); assert( Value == 4 );
        Num = 1;         Value = fwrite( &Num, 1, 4, pFileOutY ); assert( Value == 4 );

        nBytes = nSize[n] * 1 / 8;
        assert( nSize[n] * 1 % 8 == 0 );
        Value = fwrite( Vec_BitArray(vBitY), 1, nBytes, pFileOutY );
        assert( Value == nBytes );

        Vec_BitFree( vBitX );
        Vec_BitFree( vBitY );

        fclose( pFileOutX );
        fclose( pFileOutY );
    }
}

/**Function*************************************************************

  Synopsis    [Converts CIFAR-10 to 1-bit pixels and binary output {0-4} vs {5-9}.]

  Description [https://www.cs.toronto.edu/~kriz/cifar.html]
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
void Lgn_NetCifar100Convert()
{
    // The binary version of the CIFAR-100 is just like the binary version 
    // of the CIFAR-10, except that each image has two label bytes (coarse and fine) 
    // and 3072 pixel bytes, so the binary files look like this: 
    // <1 x coarse label><1 x fine label><3072 x pixel>
    char * pFileNameInT  =  "data/cifar100/train.bin";
    char * pFileNameInV  =  "data/cifar100/test.bin";

    char * pFileNameOutTX = "data/cifar100/cifar100_50k_32_32_3_1.data";
    char * pFileNameOutTY = "data/cifar100/cifar100_50k_1.data";
    char * pFileNameOutVX = "data/cifar100/cifar100_10k_32_32_3_1.data";
    char * pFileNameOutVY = "data/cifar100/cifar100_10k_1.data";
    int n, i, x, y, Num, Value, nBytes;

    int nSize[2] = {50000, 10000};
    for ( n = 0; n < 2; n++ )
    {
        FILE * pFileOutX = fopen( n ? pFileNameOutVX : pFileNameOutTX, "wb" );
        FILE * pFileOutY = fopen( n ? pFileNameOutVY : pFileNameOutTY, "wb" );

        Vec_Bit_t * vBitX = Vec_BitAlloc( nSize[n] * 32 * 32 * 3 * 1 );
        Vec_Bit_t * vBitY = Vec_BitAlloc( nSize[n] * 1 );
        FILE * pFile = fopen( n ? pFileNameInV : pFileNameInT, "rb" );
        for ( i = 0; i < nSize[n]; i++ )
        {
            unsigned char pBuffer[3100];
            Value = fread( pBuffer, 1, 3074, pFile );
            assert( Value == 3074 );
            assert( pBuffer[1] >= 0 && pBuffer[1] < 100 );
            Vec_BitPush( vBitY, (pBuffer[1] >= 50) & 1 );
            for ( y = 0; y < 32; y++ )
            for ( x = 0; x < 32; x++ )
            {
                Vec_BitPush( vBitX, (pBuffer[2+1024*0+y*32+x] >> 7) & 1 );
                Vec_BitPush( vBitX, (pBuffer[2+1024*1+y*32+x] >> 7) & 1 );
                Vec_BitPush( vBitX, (pBuffer[2+1024*2+y*32+x] >> 7) & 1 );
            }
        }
        fclose( pFile );
        assert( Vec_BitSize(vBitX) == Vec_BitCap(vBitX) );
        assert( Vec_BitSize(vBitY) <= Vec_BitCap(vBitY) );

        Num = 5;         Value = fwrite( &Num, 1, 4, pFileOutX ); assert( Value == 4 );
        Num = nSize[n];  Value = fwrite( &Num, 1, 4, pFileOutX ); assert( Value == 4 );
        Num = 32;        Value = fwrite( &Num, 1, 4, pFileOutX ); assert( Value == 4 );
        Num = 32;        Value = fwrite( &Num, 1, 4, pFileOutX ); assert( Value == 4 );
        Num = 3;         Value = fwrite( &Num, 1, 4, pFileOutX ); assert( Value == 4 );
        Num = 1;         Value = fwrite( &Num, 1, 4, pFileOutX ); assert( Value == 4 );

        nBytes = nSize[n] * 32 * 32 * 3 * 1 / 8;
        assert( nSize[n] * 32 * 32 * 3 * 1 % 8 == 0 );
        Value = fwrite( Vec_BitArray(vBitX), 1, nBytes, pFileOutX );
        assert( Value == nBytes );


        Num = 2;         Value = fwrite( &Num, 1, 4, pFileOutY ); assert( Value == 4 );
        Num = nSize[n];  Value = fwrite( &Num, 1, 4, pFileOutY ); assert( Value == 4 );
        Num = 1;         Value = fwrite( &Num, 1, 4, pFileOutY ); assert( Value == 4 );

        nBytes = nSize[n] * 1 / 8;
        assert( nSize[n] * 1 % 8 == 0 );
        Value = fwrite( Vec_BitArray(vBitY), 1, nBytes, pFileOutY );
        assert( Value == nBytes );

        Vec_BitFree( vBitX );
        Vec_BitFree( vBitY );

        fclose( pFileOutX );
        fclose( pFileOutY );
    }
}

/**Function*************************************************************

  Synopsis    [Dumps Verilog for LogicNet]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
void Lgn_PrintBinary( FILE * pFile, unsigned Sign[], int nBits )
{
    int Remainder, nWords;
    int w, i;
    Remainder = (nBits%(sizeof(unsigned)*8));
    nWords    = (nBits/(sizeof(unsigned)*8)) + (Remainder>0);
    for ( w = nWords-1; w >= 0; w-- )
        for ( i = ((w == nWords-1 && Remainder)? Remainder-1: 31); i >= 0; i-- )
            fprintf( pFile, "%c", '0' + (int)((Sign[w] & (1<<i)) > 0) );
   fprintf( pFile, "\n" );
}
void Lgn_PrintHex( FILE * pFile, word * pBits, int nBits )
{
    int k, nDigits = nBits / 4 + ((nBits % 4) > 0);
    for ( k = nDigits - 1; k >= 0; k-- )
    {
        int Digit = (int)((pBits[k/16] >> ((k%16)*4)) & 15);
        if ( Digit < 10 )
            fprintf( pFile, "%d", Digit );
        else
            fprintf( pFile, "%c", 'A' + Digit-10 );
    }
}
void Lgn_NetDumpVerilogLut( FILE * pFile, Lgn_Layer_t * pLayer )
{
    fprintf( pFile, "module layer%02d_lut%02d #( parameter TT = %d\'h0 ) ( input [%d:0] in, output out );\n", 
        pLayer->Id, pLayer->LutSize, 1 << pLayer->LutSize, pLayer->LutSize-1 );
    fprintf( pFile, "    assign out = TT[in];\n" );
    fprintf( pFile, "endmodule\n\n" );
}
void Lgn_NetDumpVerilogLayer( FILE * pFile, Lgn_Layer_t * pLayer, int nIns, int nOuts )
{
    int i, iFanin, iLut, nDigitsIn = Abc_Base10Log(nIns), nDigitsOut = Abc_Base10Log(nOuts);
    fprintf( pFile, "module layer%02d (\n", pLayer->Id );
    fprintf( pFile, "    input  [%d:0] x,\n", nIns-1 );
    if ( nOuts == 1 )
    fprintf( pFile, "    output y\n" );
    else
    fprintf( pFile, "    output [%d:0] y\n",  nOuts-1 );
    fprintf( pFile, ");\n"                      );
    for ( iLut = 0; iLut < nOuts; iLut++ )
    {
        fprintf( pFile, "    layer%02d_lut%02d #(%d\'h", pLayer->Id, pLayer->LutSize, 1 << pLayer->LutSize );
        Lgn_PrintHex( pFile, Lgn_LayerLutTruth(pLayer, iLut), 1 << pLayer->LutSize );
        fprintf( pFile, ") i%0*d ( {", nDigitsOut, iLut );
        Lgn_LutForEachFanin( pLayer, iLut, iFanin, i )
            fprintf( pFile, "x[%0*d]%s", nDigitsIn, iFanin, i == pLayer->LutSize-1 ? "":", " );
        if ( nOuts == 1 )
        fprintf( pFile, "},  y );\n" );
        else
        fprintf( pFile, "},  y[%0*d] );\n", nDigitsOut, iLut );
    }
    fprintf( pFile, "endmodule\n\n" );
}
void Lgn_NetDumpVerilog( Lgn_Net_t * p, char * pFileName )
{
    Lgn_Layer_t * pLayer; int i;
    FILE * pFile = fopen( pFileName, "wb" );
    if ( pFile == NULL )
    {
        printf( "Cannot open output file \"%s\".\n", pFileName );
        return;
    }
    fprintf( pFile, "// This file is produced by LogicNet toolbox in ABC\n\n" );
    fprintf( pFile, "module LogicNet (\n"       );
    fprintf( pFile, "    input  [%d:0] x,\n", p->DataTX.nSampleBits-1 );
    if ( p->DataTY.nSampleBits == 1 )
    fprintf( pFile, "    output y\n" );
    else
    fprintf( pFile, "    output [%d:0] y\n",  p->DataTY.nSampleBits-1 );
    fprintf( pFile, ");\n\n"                      );
    Lgn_NetForEachLayer( p, pLayer, i ) if ( i < Lgn_NetLayerNum(p)-1 )
    fprintf( pFile, "    wire [%d:0] s%02d;\n", pLayer->nLuts-1, i );
    fprintf( pFile, "\n" );
    fprintf( pFile, "    layer%02d layer%02d_inst ( x, s%02d );\n", 0, 0, 0 );
    Lgn_NetForEachLayer( p, pLayer, i ) if ( i > 0 && i < Lgn_NetLayerNum(p)-1 )
    fprintf( pFile, "    layer%02d layer%02d_inst ( s%02d, s%02d );\n", i, i, i-1, i );
    fprintf( pFile, "    layer%02d layer%02d_inst ( s%02d, y );\n", i-1, i-1, i-2 );
    fprintf( pFile, "\n" );
    fprintf( pFile, "endmodule\n\n" );
    Lgn_NetForEachLayer( p, pLayer, i )
        Lgn_NetDumpVerilogLayer( pFile, pLayer, 
            i ? Lgn_LayerPrev(pLayer)->nLuts : p->DataTX.nSampleBits,
            i < Lgn_NetLayerNum(p)-1 ? pLayer->nLuts : p->DataTY.nSampleBits );
    fprintf( pFile, "\n" );
    Lgn_NetForEachLayer( p, pLayer, i )
        Lgn_NetDumpVerilogLut( pFile, pLayer );
    fprintf( pFile, "\n" );
    fclose( pFile );
}


/**Function*************************************************************

  Synopsis    [Dumps Verilog for CubeNet]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
int Lgn_TranslateCubeGuide( int Guide, int LutSize, int pPolars[32] )
{
    int k;
    for ( k = 0; k < LutSize; k++, Guide /= 3 )
        pPolars[k] = Guide % 3;
    assert( Guide == 0 || Guide == 1 );
    return Guide;
}
void Lgn_NetDumpVerilogCube( Lgn_Net_t * p, char * pFileName, char * pPref )
{
    int fOneLine = 3;
    Lgn_Layer_t * pLayer; char * pSpot = NULL, * pStart = NULL;
    int i, k, iLut, iFanin, pPolars[32], nLayers = Lgn_NetLayerNum(p);
    FILE * pFile = fopen( pFileName, "wb" );
    if ( pFile == NULL )
    {
        printf( "Cannot open output file \"%s\".\n", pFileName );
        return;
    }
    fprintf( pFile, "// This file is produced by LogicNet toolbox in ABC\n\n" );
    pSpot = strrchr( pFileName, '.' );
    if ( pSpot != NULL )
        *pSpot = '\0';
    for ( pStart = pSpot-1; pStart != pFileName; pStart-- )
        if ( *pStart == '\\' || *pStart == '/' )
        {
            pStart++;
            break;
        }
    fprintf( pFile, "module %s (\n", pStart );
    if ( pSpot != NULL )
        *pSpot = '.';
    fprintf( pFile, "    input  [%d:0] %sx,\n", p->DataTX.nSampleBits-1, pPref );
    if ( p->DataTY.nSampleBits == 1 )
    fprintf( pFile, "    output %sy\n", pPref );
    else
    fprintf( pFile, "    output [%d:0] %sy\n", p->DataTY.nSampleBits-1, pPref );
    fprintf( pFile, ");\n\n"                      );
    for ( i = 0; i < p->DataTX.nSampleBits; i++ )
    fprintf( pFile, "    wire %ss00_%04d = %sx[%d];\n", pPref, i, pPref, i );
    fprintf( pFile, "\n" );

    Lgn_NetForEachLayer( p, pLayer, i )
    {
        Lgn_LayerForEachLut( pLayer, iLut )
        {
            int Guide = Lgn_LayerLutPolar( pLayer, iLut );
            int Phase = Lgn_TranslateCubeGuide( Guide, pLayer->LutSize, pPolars );
            if ( fOneLine == 1 )
            {
                int LastInd = -1;
                for ( k = 0; k < pLayer->LutSize; k++ ) if ( pPolars[k] != 2 )
                    LastInd = k;
                fprintf( pFile, "    wire %ss%02d_%04d =", pPref, i+1, iLut );
                Lgn_LutForEachFanin( pLayer, iLut, iFanin, k ) if ( pPolars[k] != 2 )
                    fprintf( pFile, " %s%ss%02d_%04d%s", ((pPolars[k] == 1) ^ Phase) ? "~":" ", pPref, i, iFanin, k == LastInd ? "":(Phase ? " |":" &") );
                fprintf( pFile, ";\n" );
            }
            else if ( fOneLine == 2 )
            {
                int pLits[32], nLits = 0;
                for ( k = 0; k < pLayer->LutSize; k++ )
                    if ( pPolars[k] != 2 )
                        pLits[nLits++] = Abc_Var2Lit( Vec_IntEntry(&pLayer->vFanins, iLut*pLayer->LutSize+k), ((pPolars[k] == 1) ^ Phase) );
                if ( nLits == 1 )
                    fprintf( pFile, "    wire %ss%02d_%04d   = %s%ss%02d_%04d;\n", pPref, i+1, iLut, Abc_LitIsCompl(pLits[0]) ? "~":" ", pPref, i, Abc_Lit2Var(pLits[0])  );
                else if ( nLits == 2 )
                    fprintf( pFile, "    wire %ss%02d_%04d   = %s%ss%02d_%04d   %c %s%ss%02d_%04d;\n", pPref, i+1, iLut, 
                        Abc_LitIsCompl(pLits[0]) ? "~":" ", pPref, i, Abc_Lit2Var(pLits[0]),  Phase ? '|':'&',
                        Abc_LitIsCompl(pLits[1]) ? "~":" ", pPref, i, Abc_Lit2Var(pLits[1])  );
                else
                {
                    assert( nLits > 2 );
                    fprintf( pFile, "    wire %ss%02d_%04d_0 = %s%ss%02d_%04d   %c %s%ss%02d_%04d;\n", pPref, i+1, iLut, 
                        Abc_LitIsCompl(pLits[0]) ? "~":" ", pPref, i, Abc_Lit2Var(pLits[0]),  Phase ? '|':'&',
                        Abc_LitIsCompl(pLits[1]) ? "~":" ", pPref, i, Abc_Lit2Var(pLits[1])  );
                    for ( k = 2; k < nLits-1; k++ )
                    {
                    fprintf( pFile, "    wire %ss%02d_%04d_%d = %s%ss%02d_%04d_%d %c %s%ss%02d_%04d;\n", pPref, i+1, iLut, k-1,
                                                       " ", pPref, i+1, iLut, k-2,  Phase ? '|':'&',
                        Abc_LitIsCompl(pLits[k]) ? "~":" ", pPref, i,   Abc_Lit2Var(pLits[k])  );
                    }
                    fprintf( pFile, "    wire %ss%02d_%04d   = %s%ss%02d_%04d_%d %c %s%ss%02d_%04d;\n", pPref, i+1, iLut,
                                                       " ", pPref, i+1, iLut, k-2,  Phase ? '|':'&',
                        Abc_LitIsCompl(pLits[k]) ? "~":" ", pPref, i,   Abc_Lit2Var(pLits[k])  );
                }
            }
            else if ( fOneLine == 3 ) 
            {
                int nLits = 0;
                for ( k = 0; k < pLayer->LutSize; k++ )
                    if ( pPolars[k] != 2 )
                        fprintf( pFile, "    wire %st%02d_%04d_%d = %s%ss%02d_%04d;\n", pPref, 
                            i+1, iLut, nLits++,  ((pPolars[k] == 1) ^ Phase) ? "~":" ", pPref, 
                            i,   Vec_IntEntry(&pLayer->vFanins, iLut*pLayer->LutSize+k)  );

                if ( nLits == 1 )
                    fprintf( pFile, "    wire %ss%02d_%04d   = %st%02d_%04d_0;\n", pPref, i+1, iLut, pPref, i+1, iLut );
                else if ( nLits == 2 )
                    fprintf( pFile, "    wire %ss%02d_%04d   = %st%02d_%04d_0 %c %st%02d_%04d_1;\n", pPref, 
                        i+1, iLut, pPref,  i+1, iLut,  Phase ? '|':'&', pPref,  i+1, iLut );
                else
                {
                    assert( nLits > 2 );
                    fprintf( pFile, "    wire %ss%02d_%04d_0 = %st%02d_%04d_0 %c %st%02d_%04d_1;\n", pPref, 
                        i+1, iLut, pPref,  i+1, iLut,  Phase ? '|':'&', pPref,  i+1, iLut );
                    for ( k = 2; k < nLits-1; k++ )
                    {
                    fprintf( pFile, "    wire %ss%02d_%04d_%d = %ss%02d_%04d_%d %c %st%02d_%04d_%d;\n", pPref, 
                        i+1, iLut, k-1, pPref,  i+1, iLut, k-2,  Phase ? '|':'&', pPref,  i+1, iLut, k );
                    }
                    fprintf( pFile, "    wire %ss%02d_%04d   = %ss%02d_%04d_%d %c %st%02d_%04d_%d;\n", pPref, 
                        i+1, iLut,      pPref,  i+1, iLut, k-2,  Phase ? '|':'&', pPref,  i+1, iLut, k );
                }
                fprintf( pFile, "\n" );
            }
        }
        fprintf( pFile, "\n" );
    }

    if ( p->DataTY.nSampleBits == 1 )
    fprintf( pFile, "    assign %sy = %ss%02d_%04d;\n", pPref, pPref, nLayers, 0 );
    else
    {
    fprintf( pFile, "    assign %sy = { " );
    for ( i = 0; i < p->DataTY.nSampleBits; i++ )
    fprintf( pFile, "%ss%02d_%04d%s ", pPref, pPref, nLayers, i, i == p->DataTY.nSampleBits-1 ? "":"," );
    fprintf( pFile, "};\n" );
    }
    fprintf( pFile, "\n" );
    fprintf( pFile, "endmodule\n\n" );
    fclose( pFile );
}


/**Function*************************************************************

  Synopsis    [Creates fanouts and TFO cones.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
void Lgn_NetPrintFanins( Lgn_Net_t * p )
{
    Lgn_Layer_t * pLayer; int i, k, iLut, iFanin;
    Lgn_NetForEachLayer( p, pLayer, i )
    {
        printf( "Layer %d\n", i );
        Lgn_LayerForEachLut( pLayer, iLut )
        {
            printf( "Lut %4d : Fanins : ", iLut );
            Lgn_LutForEachFanin( pLayer, iLut, iFanin, k )
                printf( "%d ", iFanin );
            printf( "\n" );
        }
    }
}
void Lgn_NetPrintFanouts( Lgn_Net_t * p )
{
    Lgn_Layer_t * pLayer; int i, k, iLut, iFanout;
    Lgn_NetForEachLayer( p, pLayer, i )
    {
        printf( "Layer %d\n", i );
        Lgn_LayerForEachLut( pLayer, iLut )
        {
            printf( "Lut %4d : Fanouts : ", iLut );
            Lgn_LutForEachFanout( pLayer, iLut, iFanout, k )
                printf( "%d(%d) ", iFanout & 0xFFFF, iFanout >> 16 );
            printf( "\n" );
        }
    }
}
void Lgn_NetPrintTfoCones( Lgn_Net_t * p )
{
    Lgn_Layer_t * pLayer; int i, k, iLut, iNode;
    Lgn_NetForEachLayer( p, pLayer, i )
    {
        printf( "Layer %d\n", i );
        Lgn_LayerForEachLut( pLayer, iLut )
        {
            printf( "Lut %4d : TFO cone : ", iLut );
            Lgn_LutForEachTfoNode( pLayer, iLut, iNode, k )
                printf( "%d(%d) ", iNode & 0xFFFF, iNode >> 16 );
            printf( "\n" );
        }
    }
}
void Lgn_NetCreateFanouts( Lgn_Net_t * p )
{
    Lgn_Layer_t * pLayer; int i;
    Lgn_NetForEachLayer( p, pLayer, i )
    {
        Vec_Int_t * vArray; int k, iFanin, iLut;
        Lgn_Layer_t * pLayerPrev = Lgn_LayerPrev(pLayer);
        if ( pLayerPrev == NULL )
            continue;
        Vec_WecInit( &pLayerPrev->vFanouts, pLayerPrev->nLuts );
        Vec_WecForEachLevel( &pLayerPrev->vFanouts, vArray, k )
            Vec_IntGrow( vArray, 2*pLayer->LutSize );
        Lgn_LayerForEachLut( pLayer, iLut )
            Lgn_LutForEachFanin( pLayer, iLut, iFanin, k )
                Vec_IntPush( Vec_WecEntry(&pLayerPrev->vFanouts, iFanin), (i << 16) | iLut );
    }
    //Lgn_NetPrintFanins( p );
    //Lgn_NetPrintFanouts( p );
}
void Lgn_NetCreateTfoCones( Lgn_Net_t * p )
{
    Lgn_Layer_t * pLayer; int i;
    Vec_Int_t * vConeA = Vec_IntAlloc( 1000 );
    Vec_Int_t * vConeB = Vec_IntAlloc( 1000 );
    Vec_Int_t * vTemp;
    assert( Lgn_NetLayerNum(p) < (1 << 16) );
    Lgn_NetForEachLayerReverse( p, pLayer, i )
    {
        Vec_Int_t * vArray, * vTfoCone; int k, iFanout, iLut;
        Lgn_Layer_t * pLayerNext = Lgn_LayerNext(pLayer);
        assert( pLayer->nLuts < (1 << 16) );
        Vec_WecInit( &pLayer->vTfoNodes, pLayer->nLuts );
        if ( i == Lgn_NetLayerNum(p) - 1 )
            continue;
        if ( i == Lgn_NetLayerNum(p) - 2 )
        {
            Lgn_LayerForEachLut( pLayer, iLut )
            {
                vArray   = Vec_WecEntry(&pLayer->vFanouts, iLut);
                vTfoCone = Vec_WecEntry(&pLayer->vTfoNodes, iLut);
                Vec_IntGrow( vTfoCone, Vec_IntSize(vArray) );
                Vec_IntAppend( vTfoCone, vArray );
            }
            continue;
        }
        Lgn_LayerForEachLut( pLayer, iLut )
        {
            Vec_IntClear( vConeA );
            Vec_IntClear( vConeB );
            Lgn_LutForEachFanout( pLayer, iLut, iFanout, k )
            {
                assert( pLayerNext->Id == (iFanout >> 16) );
                vTfoCone = Vec_WecEntry(&pLayerNext->vTfoNodes, iFanout & 0xFFFF );
                Vec_IntTwoMerge2( vTfoCone, vConeA, vConeB );
                ABC_SWAP( Vec_Int_t *, vConeA, vConeB );
            }
            vTfoCone = Vec_WecEntry( &pLayer->vTfoNodes, iLut );
            assert( Vec_IntSize(vTfoCone) == 0 );
            vTemp = Vec_WecEntry( &pLayer->vFanouts, iLut);
            Vec_IntGrow( vTfoCone, Vec_IntSize(vConeA) + Vec_IntSize(vTemp) );
            Vec_IntAppend( vTfoCone, vTemp );
            Vec_IntAppend( vTfoCone, vConeA );
        }
    }
    Vec_IntFree( vConeA );
    Vec_IntFree( vConeB );
    //Lgn_NetPrintTfoCones( p );
}

/**Function*************************************************************

  Synopsis    []

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
static inline void Lgn_Transpose64( word A[64] )
{
    int j, k;
    word t, m = 0x00000000FFFFFFFF;
    for ( j = 32; j != 0; j = j >> 1, m = m ^ (m << j) )
    {
        for ( k = 0; k < 64; k = (k + j + 1) & ~j )
        {
            t = (A[k] ^ (A[k+j] >> j)) & m;
            A[k] = A[k] ^ t;
            A[k+j] = A[k+j] ^ (t << j);
        }
    }
}
static inline void Lgn_Print64( word A[64] )
{
    int a, w;
    for ( a = 0; a < 64; a++, printf("\n") )
        for ( w = 0; w < 64; w++ )
            printf( "%d", (int)((A[a] >> w) & 1) );
    printf("\n");
}


/**Function*************************************************************

  Synopsis    [Code to randomize training labels.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
/*
unsigned char * Mnist_PermuteLabels( unsigned char * pLabels, int nLabels )
{
    unsigned char * pPerm = calloc( 1, 8+nLabels ); int i;
    for ( i = 0; i < nLabels; i++ )
        pPerm[8+(i+127)%nLabels] = pLabels[8+i];
    return pPerm;
}
unsigned char * Mnist_PermuteLabelsPartial( unsigned char * pLabels, int nLabels, int n )
{
    unsigned char * pPerm = calloc( 1, 8+nLabels ); int i;
    for ( i = 0; i < nLabels; i++ )
        pPerm[8+i] = pLabels[8+i];
    for ( i = 0; i < nLabels; i++ )
        if ( (i % n) == 1 )
            pPerm[8+(i+127)%nLabels] = pLabels[8+i];
    return pPerm;
}
*/
word * Lgn_PermuteLabels( word * pLabels, int nLabels )
{
    int i, nWords = Abc_Bit6WordNum(nLabels);
    word * pPerm = ABC_CALLOC( word, nWords ); 
    for ( i = 0; i < nLabels; i++ )
        if ( Abc_TtGetBit(pLabels, i) )
            Abc_TtXorBit(pPerm, (i+127)%nLabels );
    return pPerm;
}
// partial ranmodization (every n'th label is random, n > 0)
word * Lgn_PermuteLabelsPartial( word * pLabels, int nLabels, int n )
{
    int i, nWords = Abc_Bit6WordNum(nLabels);
    word * pPerm = ABC_CALLOC( word, nWords ); 
    memcpy( pPerm, pLabels, sizeof(word)*nWords );
    for ( i = 0; i < nLabels; i++ )
        if ( (i % n) == 0 )
            if ( Abc_TtGetBit(pPerm, (i+127)%nLabels ) != Abc_TtGetBit(pLabels, i) )
                Abc_TtXorBit(pPerm, (i+127)%nLabels );
    return pPerm;
}

/**Function*************************************************************

  Synopsis    [LogicNet alloc and free.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
Lgn_Net_t * Lgn_NetAlloc( Vec_Int_t * vSpec )
{
    Lgn_Net_t * p = ABC_CALLOC( Lgn_Net_t, 1 );
    Lgn_Layer_t * pLayer; 
    int i, nLayers = Vec_IntSize(vSpec)/2;
    Vec_PtrGrow( &p->vLayers, nLayers );
    for ( i = 0; i < nLayers; i++ )
        Vec_PtrPush( &p->vLayers, ABC_CALLOC(Lgn_Layer_t, 1) );
    Lgn_NetForEachLayer( p, pLayer, i )
    {
        pLayer->Id          = i;
        pLayer->pNet        = p;
        pLayer->nLuts       = Vec_IntEntry(vSpec, 2*i);
        pLayer->LutSize     = Vec_IntEntry(vSpec, 2*i+1);
        pLayer->nTruthWords = Abc_Bit6WordNum(1 << pLayer->LutSize);
    }
    return p;
}
void Lgn_NetFree( Lgn_Net_t * p )
{
    Lgn_Layer_t * pLayer; int i;
    Lgn_NetForEachLayer( p, pLayer, i )
    {
        Vec_IntErase( &pLayer->vMark );
        Vec_IntErase( &pLayer->vFanins );
        Vec_IntErase( &pLayer->vPolars );
        Vec_WecErase( &pLayer->vFanouts );
        Vec_WecErase( &pLayer->vTfoNodes );
        Vec_WrdErase( &pLayer->vTruths );
        Vec_WrdErase( &pLayer->vSims );
        Vec_WrdErase( &pLayer->vCare );
        ABC_FREE( pLayer );
    }
    Vec_PtrErase( &p->vLayers );
    Vec_WrdErase( &p->vSimsPi );
    for ( i = 0; i < p->DataTY.nSampleBits; i++ )
        Vec_StrErase( &p->vOutValue[i] );
    ABC_FREE( p->vOutValue );
    for ( i = 0; i < p->DataTY.nSampleBits; i++ )
        ABC_FREE( p->pOutValue[i] );
    ABC_FREE( p->pOutValue );
    ABC_FREE( p->DataTX.pData );
    ABC_FREE( p->DataTY.pData );
    ABC_FREE( p->DataVX.pData );
    ABC_FREE( p->DataVY.pData );
    ABC_FREE( p );
}


/**Function*************************************************************

  Synopsis    [Loads one data matrix (input or output; training or validation)]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
int Lgn_DataLoad( Lgn_Data_t * p, char * pFileName )
{
    int d, * pData = (int *)Abc_FileReadContents( pFileName, NULL );
    if ( pData == NULL )
    {
        printf( "Cannot load data from file \"%s\".\n", pFileName );
        return 0;
    }
    // the number of dimensions written in the first integer
    p->nDims       = pData[0];
    // the number of data samples
    p->nSamples    = pData[1];
    // the size of one-bit-per-sample data in words
    p->nDataWords  = Abc_Bit6WordNum(p->nSamples);
    // the number of bits in one data sample
    p->nSampleBits = 1;
    for ( d = 0; d < p->nDims-1; d++ )
        p->nSampleBits *= pData[2+d];
    // save data in the buffer aligned at 64-bit boundary
    p->pData = ABC_CALLOC( word, Abc_Bit6WordNum(p->nSamples * p->nSampleBits) );
    memcpy( p->pData, pData + 1 + p->nDims, Abc_BitByteNum(p->nSamples * p->nSampleBits) );
    ABC_FREE( pData );
    fflush( stdout );
#ifdef LGN_VERBOSE
    printf( "Finished entering data from file  %-48s with %6d samples and %6d bits per sample.\n", 
        pFileName, p->nSamples, p->nSampleBits );
#endif
    return 1;
}

/**Function*************************************************************

  Synopsis    [Implication checking.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
// computes implications for each TT minterm
int Lgn_NetImplications( word * pTruth, int LutSize, int iMint )
{
    word pQuant[1024], pCof0[1024], pCof1[1024], pSpot[1024];
    int nWords = Abc_Truth6WordNum( LutSize );
    int Value  = Abc_TtGetBit( pTruth, iMint );
    int v, nMints = 1 << LutSize, ValueImpls = 0;
    assert( iMint <= nMints );
    assert( LutSize <= 16 );
    Abc_TtClear( pSpot, nWords );
    Abc_TtSetBit( pSpot, iMint );
    for ( v = 0; v < LutSize; v++ )
    {
        Abc_TtCofactor0p( pCof0, pSpot, nWords, v );
        Abc_TtCofactor1p( pCof1, pSpot, nWords, v );
        Abc_TtOr( pQuant, pCof0, pCof1, nWords );
        if ( Abc_TtIntersect( pTruth, pQuant, nWords, Value ) )
            ValueImpls |= 1 << v;
        else
            Abc_TtCopy( pSpot, pQuant, nWords, 0 );
    }
    return ValueImpls;
}
void Lgn_NetImplicationsTest()
{
    word Truth = (s_Truths6[0] & s_Truths6[1]) | (s_Truths6[2] & s_Truths6[3]);
    int m;
    for ( m = 0; m < 16; m++ )
    {
        int ValueImpls = Lgn_NetImplications( &Truth, 4, m );
        printf( "%2d : %2d ", m, ValueImpls );
        Lgn_PrintBinary( stdout, (unsigned *)&ValueImpls, 4 );
    }
}

/**Function*************************************************************

  Synopsis    [LogicNet training.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
// creates random fanins and truth tables for all layers
void Lgn_NetPrepare( Lgn_Net_t * p )
{
    Lgn_Layer_t * pLayer; 
    int i, k, iLut, iFanin, nRandomSeed = 3;
    int nMulti = p->DataTY.nSampleBits;
    Vec_Int_t * vFanins = Vec_IntAlloc( 100 );
    Abc_Random(1);
    for ( i = 0; i < nRandomSeed; i++ )
        Abc_Random(0);
    Lgn_NetForEachLayer( p, pLayer, i )
    {
        int nItems, nInputBits = i ? Lgn_NetLayer(p, i-1)->nLuts : p->DataTX.nSampleBits;

        // clean fanins
        //Vec_IntFill( &pLayer->vFanins,  (nItems = pLayer->nLuts*pLayer->LutSize),     0 );
        // assign random fanins
        Vec_IntGrow( &pLayer->vFanins, (nItems = pLayer->nLuts*pLayer->LutSize) );
        //for ( k = 0; k < nItems; k++ ) 
        //    Vec_IntPush( &pLayer->vFanins, Abc_Random(0) % nInputBits ); 
        // make sure fanins are unique and ordered
        for ( k = 0; k < pLayer->nLuts; k++ ) 
        {
            Vec_IntClear( vFanins );
            while ( Vec_IntSize(vFanins) < pLayer->LutSize )
                //Vec_IntPushUniqueOrder( vFanins, (Abc_Random(0) * nMulti + (k & 1) * (k % nMulti)) % nInputBits );
                Vec_IntPushUniqueOrder( vFanins, (Abc_Random(0) * nMulti + k % nMulti) % nInputBits );
            Vec_IntAppend( &pLayer->vFanins, vFanins );
        }

        // clean polarities
        Vec_IntFill( &pLayer->vPolars, pLayer->nLuts, -1 );

        // clean truth tables
        //Vec_WrdFill( &pLayer->vTruths,  (nItems = pLayer->nLuts*pLayer->nTruthWords), 0 );
        // assign random truth tables
        Vec_WrdGrow( &pLayer->vTruths, (nItems = pLayer->nLuts*pLayer->nTruthWords) );
        for ( k = 0; k < nItems; k++ ) 
            Vec_WrdPush( &pLayer->vTruths, Abc_RandomW(0) ); 
    }
    Vec_IntFree( vFanins );
    // mark used LUTs
    pLayer = Lgn_NetLayerLast( p );
    Vec_IntFill( &pLayer->vMark, pLayer->nLuts, 1 );
    Lgn_NetForEachLayerReverse( p, pLayer, i )
    {
        Lgn_Layer_t * pLayerPrev = Lgn_LayerPrev(pLayer);
        if ( pLayerPrev == NULL )
            continue;
        Vec_IntFill( &pLayerPrev->vMark, pLayerPrev->nLuts, 0 );
        Lgn_LayerForEachLut( pLayer, iLut )
            if ( Vec_IntEntry(&pLayer->vMark, iLut) )
                Lgn_LutForEachFanin( pLayer, iLut, iFanin, k )
                    Vec_IntWriteEntry( &pLayerPrev->vMark, iFanin, 1 );
    }
}
// assigns sim info of the first layer using the input data matrix
void Lgn_NetStartSimInfo( Lgn_Net_t * p, int fValidation )
{
    Lgn_Layer_t * pLayer; int i, b, s;
    Lgn_Data_t * pDataIn = fValidation ? &p->DataVX : &p->DataTX;
    p->nSimWords = pDataIn->nDataWords;
    Lgn_NetForEachLayer( p, pLayer, i )
        Vec_WrdFill( &pLayer->vSims, pLayer->nLuts*p->nSimWords, 0 );
    Vec_WrdFill( &p->vSimsPi, pDataIn->nSampleBits*p->nSimWords, 0 ); // input sim info
    for ( b = 0; b < pDataIn->nSampleBits; b++ )
    {
        word * pSims = Lgn_NetSimsIn( p, b );
        for ( s = 0; s < pDataIn->nSamples; s++ )
            if ( Lgn_DataBit( pDataIn, s, b ) )
                Abc_TtSetBit( pSims, s );
    }
    p->vOutValue = ABC_CALLOC( Vec_Str_t, p->DataTY.nSampleBits );
    p->pOutValue = ABC_CALLOC( word *, p->DataTY.nSampleBits );
    for ( i = 0; i < p->DataTY.nSampleBits; i++ )
    {
        Vec_StrGrow( &p->vOutValue[i], pDataIn->nSamples );
        p->pOutValue[i] = ABC_CALLOC( word, p->nSimWords );
        for ( s = 0; s < pDataIn->nSamples; s++ )
        {
            Vec_StrPush( &p->vOutValue[i], (char)Lgn_DataBit(&p->DataTY, s, i) );
            if ( Lgn_DataBit(&p->DataTY, s, i) )
                Abc_TtXorBit( p->pOutValue[i], s );
        }
    }
}
// returns accuracy percentage for the output of the network
float Lgn_NetAccuracy( Lgn_Net_t * p, int fValidation )
{
    int fCountFirst = 1;
    int s, k, iLut, nCorrect  = 0;
    Lgn_Layer_t * pLayer   = Lgn_NetLayerLast(p);
    Lgn_Data_t  * pDataOut = fValidation ? &p->DataVY : &p->DataTY;
    word        * pSimsOut = Lgn_LayerLutSims( pLayer, 0 );
    assert( pLayer->nLuts == pDataOut->nSampleBits ); 
#ifdef LGN_VERBOSE
    printf( "The results of %s:\n", fValidation ? "validation" : "training" );
#endif
    for ( s = 0; s < pDataOut->nSamples; s++ )
    {
//        Lgn_LayerForEachLut( pLayer, iLut )
//            if ( Abc_TtGetBit(Lgn_LayerLutSims(pLayer, iLut), s) != Lgn_DataBit(pDataOut, s, iLut) )
//                break;
        Lgn_LayerForEachLut( pLayer, iLut )
        {
            if ( Abc_TtGetBit(Lgn_LayerLutSims(pLayer, iLut), s) != Lgn_DataBit(pDataOut, s, iLut) )
                break;
            if ( fCountFirst && Abc_TtGetBit(Lgn_LayerLutSims(pLayer, iLut), s) && Lgn_DataBit(pDataOut, s, iLut) )
            {
                iLut = pLayer->nLuts;
                break;
            }
        }
        nCorrect += (int)(iLut == pLayer->nLuts);
        if ( s >= 10 )
            continue;
#ifdef LGN_VERBOSE
        printf( "sample %3d (out of %6d)  value ", s, pDataOut->nSamples );
        Lgn_LayerForEachLut( pLayer, iLut )
            printf( "%d", Abc_TtGetBit(Lgn_LayerLutSims(pLayer, iLut), s) );
        printf( "  golden " ); 
        Lgn_LayerForEachLut( pLayer, iLut )
            printf( "%d", Lgn_DataBit(pDataOut, s, iLut) );
        printf( "  error " ); 
        Lgn_LayerForEachLut( pLayer, iLut )
        {
            printf( "%c", (Abc_TtGetBit(Lgn_LayerLutSims(pLayer, iLut), s) ^ Lgn_DataBit(pDataOut, s, iLut)) ? 'x' : '.' );
            if ( fCountFirst && Abc_TtGetBit(Lgn_LayerLutSims(pLayer, iLut), s) && Lgn_DataBit(pDataOut, s, iLut) )
            {
                for ( k = iLut+1; k < pLayer->nLuts; k++ )
                    printf( "%c", '.' );
                break;
            }
        }
        printf( "\n" ); 
        fflush( stdout );
#endif
    }
    return (float)100.0*nCorrect/pDataOut->nSamples;
}

// collect distribution of local minterms at the inputs of one LUT whose fanins are given in pFaninsSims
void Lgn_NetCountMintermsNaive( Lgn_Layer_t * pLayer, int iLut, word ** pFaninSims, int * pPatCounts0, int * pPatCounts1, Vec_Int_t * vMints )
{
    Lgn_Net_t * p = pLayer->pNet; int s;
    int Index = iLut % p->DataTY.nSampleBits;
    memset( pPatCounts0, 0, sizeof(int)*(1 << pLayer->LutSize) );
    memset( pPatCounts1, 0, sizeof(int)*(1 << pLayer->LutSize) );
    Vec_IntClear( vMints );
    for ( s = 0; s < p->DataTX.nSamples; s++ )
    {
        int k, Mint = 0;
        for ( k = 0; k < pLayer->LutSize; k++ )
            if ( Abc_TtGetBit(pFaninSims[k], s) )
                Mint |= 1 << k;
        Vec_IntPush( vMints, Mint );
        //if ( Lgn_DataBit(&p->DataTY, s, 0) )
        if ( Vec_StrEntry(&p->vOutValue[Index], s) )
            pPatCounts1[Mint] += p->DataTY.nSampleBits;
        else
            pPatCounts0[Mint]++;
    }
}
void Lgn_NetCountMintermsSmart( Lgn_Layer_t * pLayer, int iLut, word ** pFaninSims, int * pPatCounts0, int * pPatCounts1, Vec_Int_t * vMints )
{
    Lgn_Net_t * p = pLayer->pNet; word A[64]; 
    int Index = iLut % p->DataTY.nSampleBits;
    int BatchSize = pLayer->LutSize <= 8 ? 8 : 4; // one 64-bit word holds 8x 8-bit minterms or 4x 16-bit minterms
    int a, k, b, w, nWordBatches = p->nSimWords / BatchSize + (int)(p->nSimWords % BatchSize > 0);
    int nAddOn = 64 / BatchSize - pLayer->LutSize, Count = 0;
    memset( pPatCounts0, 0, sizeof(int)*(1 << pLayer->LutSize) );
    memset( pPatCounts1, 0, sizeof(int)*(1 << pLayer->LutSize) );
    Vec_IntClear( vMints );
    for ( b = 0; b < nWordBatches; b++ )
    {
        for ( a = w = 0; w < BatchSize; w++ )
        {
            for ( k = 0; k < pLayer->LutSize; k++ )
                A[63-a++] = b*BatchSize+w < p->nSimWords ? pFaninSims[k][b*BatchSize+w] : 0;
            for ( k = 0; k < nAddOn; k++ )
                A[63-a++] = 0;
        }
        assert( a == 64 );
        //Lgn_Print64( A );
        Lgn_Transpose64( A );
        //Lgn_Print64( A );
        for ( w = 0; w < BatchSize; w++ )
            for ( a = 0; a < 64; a++, Count++ )
            {
                int Mint = BatchSize == 8 ? ((unsigned char *)(&A[63-a]))[w] : ((unsigned short *)(&A[63-a]))[w];
                if ( Count >= p->DataTX.nSamples )
                    continue;
                assert( Mint >= 0 && Mint < (1 << pLayer->LutSize) );
                Vec_IntPush( vMints, Mint );
                //if ( Lgn_DataBit(&p->DataTY, Count, 0) )
                if ( Vec_StrEntry(&p->vOutValue[Index], Count) )
                    pPatCounts1[Mint] += p->DataTY.nSampleBits;
                else
                    pPatCounts0[Mint]++;
            }
    }
}
// the top-level training procedure
float Lgn_NetTrain( Lgn_Net_t * p, char * pFileNameX, char * pFileNameY, int nRandom )
{
    float Result = 0;
    Lgn_Layer_t * pLayer; int i;
    if ( !Lgn_DataLoad(&p->DataTX, pFileNameX) )
        return 0;
    if ( !Lgn_DataLoad(&p->DataTY, pFileNameY) )
        return 0;
    if ( nRandom > 0 )
    {
        // randomize training labels
        word * pDataTemp;
        assert( p->DataTY.nSampleBits == 1 );
        //p->DataTY.pData = Lgn_PermuteLabels( pDataTemp = p->DataTY.pData, p->DataTY.nSamples );
        p->DataTY.pData = Lgn_PermuteLabelsPartial( pDataTemp = p->DataTY.pData, p->DataTY.nSamples, nRandom );
        ABC_FREE( pDataTemp ); 
        printf( "Randomly permuting labels with parameter nRandom = %d.\n", nRandom );
    }
    //assert( p->DataTY.nSampleBits == 1 );               // binary classification
    assert( p->DataTX.nSamples == p->DataTY.nSamples ); // data constistency
    Lgn_NetPrepare( p );
    Lgn_NetStartSimInfo( p, 0 );
    Lgn_NetForEachLayer( p, pLayer, i )
    {
        int s, k, iLut, iFanin, nCubes;
        int nThreshold     = p->DataTX.nSamples / 10000; // 0.01%
        int Mint, nMints   = 1 << pLayer->LutSize;
        int * pPatCounts0  = ABC_CALLOC( int, nMints );
        int * pPatCounts1  = ABC_CALLOC( int, nMints );
        word * pTruth0     = ABC_CALLOC( word, pLayer->nTruthWords );
        word * pTruth1     = ABC_CALLOC( word, pLayer->nTruthWords );
        Vec_Int_t * vMints = Vec_IntAlloc( p->DataTY.nSamples );
        word ** pFaninSims = ABC_CALLOC( word *, pLayer->LutSize );
        Vec_Int_t * vFLits = Vec_IntAlloc( pLayer->LutSize );
        Vec_Int_t * vCover = Vec_IntAlloc( 1000 );
        assert( pLayer->LutSize <= 20 );
        Lgn_LayerForEachLut( pLayer, iLut )
        {
            word * pTruth   = Lgn_LayerLutTruth( pLayer, iLut );
            word * pSimsOut = Lgn_LayerLutSims( pLayer, iLut );
            // collect fanins of this LUT
            Lgn_Layer_t * pLayerPrev = Lgn_LayerPrev(pLayer);
            Lgn_LutForEachFanin( pLayer, iLut, iFanin, k )
                pFaninSims[k] = i ? Lgn_LayerLutSims(pLayerPrev, iFanin) : Lgn_NetSimsIn(p, iFanin);

            if ( 1 ) // use transpose 
                Lgn_NetCountMintermsSmart( pLayer, iLut, pFaninSims, pPatCounts0, pPatCounts1, vMints );
            else // do not use transpose
                Lgn_NetCountMintermsNaive( pLayer, iLut, pFaninSims, pPatCounts0, pPatCounts1, vMints );
/*
            // see what local minterms look like
            if ( i == 0 && (iLut == 0 || iLut == 50) )
            {
                Vec_IntForEachEntryStop( vMints, iFanin, k, 100 )
                    printf( "%d ", iFanin );
                printf( "...\n" );
            }
*/
            // compute local function of this LUT
            if ( 0 ) // use ISOP (works up to 8 inputs)
            {
                Abc_TtClear( pTruth, pLayer->nTruthWords );
                Abc_TtClear( pTruth0, pLayer->nTruthWords );
                Abc_TtClear( pTruth1, pLayer->nTruthWords );
                for ( Mint = 0; Mint < nMints; Mint++ )
                    if ( pPatCounts0[Mint] > pPatCounts1[Mint] + nThreshold )
                        Abc_TtSetBit( pTruth0, Mint );
                    else if ( pPatCounts1[Mint] > pPatCounts0[Mint] + nThreshold )
                        Abc_TtSetBit( pTruth1, Mint );
                Abc_TtNot( pTruth0, pLayer->nTruthWords );
                if ( pLayer->LutSize <= 6 )
                    pTruth[0] = Abc_Tt6Isop( pTruth1[0], pTruth0[0], pLayer->LutSize, &nCubes );
                else if ( pLayer->LutSize == 7 )
                    Abc_Tt7Isop( pTruth1, pTruth0, pLayer->LutSize, pTruth );
                else if ( pLayer->LutSize == 8 )
                    Abc_Tt8Isop( pTruth1, pTruth0, pLayer->LutSize, pTruth );
                else assert( 0 );
            }
            else // do not use ISOP
            {
                // optionally clean randomly assigned truth table
                //Abc_TtClear( pTruth, pLayer->nTruthWords );
                for ( Mint = 0; Mint < nMints; Mint++ )
                    if ( pPatCounts0[Mint] > pPatCounts1[Mint] + nThreshold ) // should be 0
                    {
                        if ( Abc_TtGetBit( pTruth, Mint ) != 0 ) // change if not 0
                            Abc_TtXorBit( pTruth, Mint );
                    }
                    else if ( pPatCounts1[Mint] > pPatCounts0[Mint] + nThreshold ) // should be 1
                    {
                        if ( Abc_TtGetBit( pTruth, Mint ) != 1 ) // change if not 1
                            Abc_TtXorBit( pTruth, Mint );
                    }
            }
            // compute LUT output values using newly derived truth table
            assert( Vec_IntSize(vMints) == p->DataTX.nSamples );
            Vec_IntForEachEntry( vMints, Mint, s )
                if ( Abc_TtGetBit(pTruth, Mint) )
                    Abc_TtSetBit( pSimsOut, s );
        }
        Vec_IntFree( vFLits );
        Vec_IntFree( vCover );
        Vec_IntFree( vMints );
        ABC_FREE( pFaninSims );
        ABC_FREE( pPatCounts0 );
        ABC_FREE( pPatCounts1 );
        ABC_FREE( pTruth0 );
        ABC_FREE( pTruth1 );
#ifdef LGN_VERBOSE
        printf( "Finished training layer %2d (out of %2d)...\r", i, Lgn_NetLayerNum(p) );
        fflush( stdout );
#endif
    }
#ifdef LGN_VERBOSE
    printf( "Finished training %2d layers.                         \n", Lgn_NetLayerNum(p) );
    fflush( stdout );
#endif
    return Lgn_NetAccuracy( p, 0 );
}


/**Function*************************************************************

  Synopsis    [LogicNet validation.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
float Lgn_NetValidate( Lgn_Net_t * p, char * pFileNameX, char * pFileNameY )
{
    float Result = 0;
    Lgn_Layer_t * pLayer; int i;
    if ( !Lgn_DataLoad(&p->DataVX, pFileNameX) )
        return 0;
    if ( !Lgn_DataLoad(&p->DataVY, pFileNameY) )
        return 0;
    assert( p->DataTX.nSamples == p->DataTY.nSamples ); // data consistency
    Lgn_NetStartSimInfo( p, 1 );
    Lgn_NetForEachLayer( p, pLayer, i )
    {
        int s, k, iLut, iFanin, nMints = 1 << pLayer->LutSize;
        word ** pFaninSims = ABC_CALLOC( word *, pLayer->LutSize );
        Lgn_Layer_t * pLayerPrev = Lgn_LayerPrev(pLayer);
        Lgn_LayerForEachLut( pLayer, iLut )
        {
            word * pTruth   = Lgn_LayerLutTruth( pLayer, iLut );
            word * pSimsOut = Lgn_LayerLutSims( pLayer, iLut );
            Lgn_LutForEachFanin( pLayer, iLut, iFanin, k )
                pFaninSims[k] = i ? Lgn_LayerLutSims(pLayerPrev, iFanin) : Lgn_NetSimsIn(p, iFanin);
            assert( Abc_TtIsConst0(pSimsOut, p->nSimWords) );
            for ( s = 0; s < p->DataVX.nSamples; s++ )
            {
                int Mint = 0; 
                for ( k = 0; k < pLayer->LutSize; k++ )
                    if ( Abc_TtGetBit(pFaninSims[k], s) )
                        Mint |= 1 << k;
                if ( Abc_TtGetBit(pTruth, Mint) )
                    Abc_TtSetBit( pSimsOut, s );
            }
        }
        ABC_FREE( pFaninSims );
#ifdef LGN_VERBOSE
        printf( "Finished validating layer %2d (out of %2d)...\r", i, Lgn_NetLayerNum(p) );
        fflush( stdout );
#endif
    }
#ifdef LGN_VERBOSE
    printf( "Finished validating %2d layers.                         \n", Lgn_NetLayerNum(p) );
    fflush( stdout );
#endif
    return Lgn_NetAccuracy( p, 1 );
}


/**Function*************************************************************

  Synopsis    [Multi-output training.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
// generates error vector for the first output
word * Lgn_NetSimErrors( Lgn_Net_t * p )
{
    Lgn_Layer_t * pLayer = Lgn_NetLayerLast(p); 
    word * pSimsOut = Lgn_LayerLutSims( pLayer, 0 );
    word * pSimsErr = ABC_CALLOC( word, p->nSimWords );
    Lgn_Data_t * pDataOut = &p->DataTY;
    int s;
    for ( s = 0; s < pDataOut->nSamples; s++ )
        if ( Abc_TtGetBit(pSimsOut, s) ^ Lgn_DataBit(pDataOut, s, 0) )
            Abc_TtXorBit(pSimsErr, s);
    return pSimsErr;
}

// count errors in the simulation data
int Lgn_NetCountErrors( Lgn_Net_t * p )
{
    int s, iLut, nErrors  = 0;
    Lgn_Data_t * pDataOut = &p->DataTY;
    Lgn_Layer_t * pLayer  = Lgn_NetLayerLast(p); 
    //assert( 64*p->nSimWords == pDataOut->nSamples );
    assert( pLayer->nLuts == pDataOut->nSampleBits );
    Lgn_LayerForEachLut( pLayer, iLut )
    {
        word * pSimsOut = Lgn_LayerLutSims( pLayer, iLut );
        //printf( "LUT %2d : ", iLut );
        for ( s = 0; s < pDataOut->nSamples; s++ )
        {
            nErrors += Abc_TtGetBit(pSimsOut, s) ^ Lgn_DataBit(pDataOut, s, iLut);
            //printf( "%d", Abc_TtGetBit(pSimsOut, s) ^ Lgn_DataBit(pDataOut, s, iLut) );
        }
        //printf( "\n" );
    }
    return nErrors;
}
// resimulate one LUT and count errors
void Lgn_LutSimulate( Lgn_Layer_t * pLayer, int iLut, Vec_Int_t * vMints, Vec_Int_t * vFlips )
{
    Lgn_Net_t * p = pLayer->pNet;
    Lgn_Data_t * pDataOut = &p->DataTY;
    word * pFaninSims[16]; int k, s, iFanin;
    word * pTruth   = Lgn_LayerLutTruth( pLayer, iLut );
    word * pSimsOut = Lgn_LayerLutSims( pLayer, iLut );
    Lgn_Layer_t * pLayerPrev = Lgn_LayerPrev(pLayer);
    assert( pLayer->LutSize <= 16 );
    assert( vMints == NULL || Vec_IntSize(vMints) == 64*p->nSimWords );
    assert( vFlips == NULL || Vec_IntSize(vFlips) == 2 << pLayer->LutSize );
    Lgn_LutForEachFanin( pLayer, iLut, iFanin, k )
        pFaninSims[k] = pLayerPrev ? Lgn_LayerLutSims(pLayerPrev, iFanin) : Lgn_NetSimsIn(p, iFanin);
    for ( s = 0; s < pDataOut->nSamples; s++ )
    {
        int Mint = 0, ValueCur = Abc_TtGetBit( pSimsOut, s ); 
        for ( k = 0; k < pLayer->LutSize; k++ )
            if ( Abc_TtGetBit(pFaninSims[k], s) )
                Mint |= 1 << k;
        if ( ValueCur == Abc_TtGetBit( pTruth, Mint ) )
            continue;
        Abc_TtXorBit( pSimsOut, s );
        if ( !Lgn_LayerIsLast(pLayer) || vFlips == NULL )
            continue;
        assert( pLayer->nLuts == pDataOut->nSampleBits );
        if ( ValueCur == Lgn_DataBit( pDataOut, s, iLut ) ) // correct -> error
            Vec_IntAddToEntry( vFlips, 2*Vec_IntEntry(vMints, s)+1, 1 ); 
        else // error -> correct
            Vec_IntAddToEntry( vFlips, 2*Vec_IntEntry(vMints, s), 1 );
    }
}
float Lgn_NetBackPropResimulate( Lgn_Net_t * p, char * pFileNameX, char * pFileNameY )
{
    float Result = 0;
    int i, iLut, nErrorsBefore, Iter, IterMax = 8;
    Lgn_Layer_t * pLayer; 
    Vec_Int_t * vMints, * vFlips;
    if ( !Lgn_DataLoad(&p->DataTX, pFileNameX) )
        return 0;
    if ( !Lgn_DataLoad(&p->DataTY, pFileNameY) )
        return 0;
    assert( p->DataTX.nSamples == p->DataTY.nSamples );            // data constistency
    assert( Lgn_NetLayerLast(p)->nLuts == p->DataTY.nSampleBits ); // data constistency
    Lgn_NetPrepare( p );
    Lgn_NetCreateFanouts( p );
    Lgn_NetCreateTfoCones( p );
    Lgn_NetStartSimInfo( p, 0 );
    Lgn_NetForEachLayer( p, pLayer, i )
        Lgn_LayerForEachLut( pLayer, iLut )
            Lgn_LutSimulate( pLayer, iLut, NULL, NULL );
    nErrorsBefore = Lgn_NetCountErrors(p);
    printf( "Original  errors = %5d (out of %5d)\n", nErrorsBefore, p->DataTY.nSamples * p->DataTY.nSampleBits ); fflush( stdout );
    vMints = Vec_IntAlloc( 64*p->nSimWords );
    vFlips = Vec_IntAlloc( 2 << pLayer->LutSize );
    for ( Iter = 0; Iter < IterMax; Iter++ )
    {
        clock_t clkStart = clock();
        //int Threshold = IterMax - Iter;
        int Threshold = Iter > 2 ? (Iter & 1) : 1;
        Lgn_NetForEachLayerReverse( p, pLayer, i )
        {
            int fChange, nLutUsed = 0, nLutChanged = 0, nMintTotal = 0, nMintCare = 0, nMintChanged = 0;
            Lgn_LayerForEachLut( pLayer, iLut )
            {
                Vec_Int_t * vTfo = Vec_WecEntry( &pLayer->vTfoNodes, iLut );
                word * pSims     = Lgn_LayerLutSims( pLayer, iLut );
                word * pTruth    = Lgn_LayerLutTruth( pLayer, iLut );
                int k, s, iNode, nErrorsAfter, Good, Bad, Gain = 0;
                word * pFaninSims[16]; 
                Lgn_Layer_t * pLayerPrev = Lgn_LayerPrev(pLayer);
                Lgn_LutForEachFanin( pLayer, iLut, iNode, k )
                    pFaninSims[k] = pLayerPrev ? Lgn_LayerLutSims(pLayerPrev, iNode) : Lgn_NetSimsIn(p, iNode);
                // collect minterms
                Vec_IntClear( vMints );
                for ( s = 0; s < 64*p->nSimWords; s++ )
                {
                    int Mint = 0; 
                    for ( k = 0; k < pLayer->LutSize; k++ )
                        if ( Abc_TtGetBit(pFaninSims[k], s) )
                            Mint |= 1 << k;
                    Vec_IntPush( vMints, Mint );
                }
                //printf( "%d ", Vec_IntSize(vTfo) );
                // simulate change
                Vec_IntFill( vFlips, 2 << pLayer->LutSize, 0 );
                Abc_TtNot( pSims, p->nSimWords );
                Vec_IntForEachEntry( vTfo, iNode, k )
                    Lgn_LutSimulate( Lgn_NetLayer(p, iNode >> 16), iNode & 0xFFFF, vMints, vFlips );
                Abc_TtNot( pSims, p->nSimWords );
                //Lgn_PrintBinary( stdout, (unsigned *)pTruth, 16 );
                // accept changes with more than threshold
                fChange = 0;
                nMintTotal += (1 << pLayer->LutSize);
                Vec_IntForEachEntryDouble( vFlips, Good, Bad, k )
                {
                    nMintCare += (int)(Good > 0 || Bad > 0);
                    if ( Good && Good >= Bad + Threshold )
                    {
                        Abc_TtXorBit( pTruth, k/2 );
                        Gain += Good - Bad;
                        nMintChanged++;
                        fChange = 1;
                    }
                }
                nLutUsed += Vec_IntEntry(&pLayer->vMark, iLut);
                nLutChanged += fChange;
                // update simulation info and resimulate
                Lgn_LutSimulate( pLayer, iLut, NULL, NULL );
                Vec_IntForEachEntry( vTfo, iNode, k )
                    Lgn_LutSimulate( Lgn_NetLayer(p, iNode >> 16), iNode & 0xFFFF, NULL, NULL );
                nErrorsAfter = Lgn_NetCountErrors(p);
                assert( nErrorsBefore - nErrorsAfter == Gain );
                nErrorsBefore = nErrorsAfter;
            }
            printf( "Training iteration %d (threshold %d):  ", Iter, Threshold );
            printf( "Level %2d  Errors = %5d (%6.2f %%)\n", i, nErrorsBefore,  100.0*nErrorsBefore/(p->DataTY.nSamples * p->DataTY.nSampleBits) );
            printf( "LUTs    : Total = %6d.  Used = %6d (%6.2f %%).  Changed = %6d (%6.2f %%).\n", 
                pLayer->nLuts, nLutUsed, 100.0*nLutUsed/pLayer->nLuts, nLutChanged,  100.0*nLutChanged/pLayer->nLuts ); 
            printf( "Minterms: Total = %6d.  Care = %6d (%6.2f %%).  Changed = %6d (%6.2f %%).\n", 
                nMintTotal,    nMintCare, 100.0*nMintCare/nMintTotal,    nMintChanged, 100.0*nMintChanged/nMintTotal );
            printf( "\n" );
            fflush( stdout );
        }
        printf( "Training iteration %d (threshold %d):  ", Iter, Threshold );
        printf( "Resulting errors = %5d  ", nErrorsBefore ); fflush( stdout );
        printf( "Time = %6.2f sec\n", (float)(clock() - clkStart)/CLOCKS_PER_SEC );
    }
    Vec_IntFree( vMints );
    Vec_IntFree( vFlips );
    return Lgn_NetAccuracy( p, 0 );
}

/**Function*************************************************************

  Synopsis    [Propagating cares.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
float Lgn_NetBackPropSensitize( Lgn_Net_t * p, char * pFileNameX, char * pFileNameY )
{
    float Result = 0;
    float ChangeThresh = (float)0.01;
    int i, iLut, nErrors, Threshold, Iter, IterMax = 4;
    Vec_Int_t * vImpls, * vGrads;
    word * pSimErrs = NULL;
    Lgn_Layer_t * pLayer; 
    if ( !Lgn_DataLoad(&p->DataTX, pFileNameX) )
        return 0;
    if ( !Lgn_DataLoad(&p->DataTY, pFileNameY) )
        return 0;
    assert( p->DataTX.nSamples == p->DataTY.nSamples );            // data constistency
    assert( Lgn_NetLayerLast(p)->nLuts == p->DataTY.nSampleBits ); // data constistency
    //assert( p->DataTY.nSampleBits == 1 ); // temporary
    Threshold = (int)(ChangeThresh * p->DataTY.nSamples);
    Lgn_NetPrepare( p );
    Lgn_NetStartSimInfo( p, 0 );
    Lgn_NetForEachLayer( p, pLayer, i )
        Lgn_LayerForEachLut( pLayer, iLut )
            Lgn_LutSimulate( pLayer, iLut, NULL, NULL );
    pSimErrs = Lgn_NetSimErrors( p );
    nErrors  = Lgn_NetCountErrors(p);
    printf( "Original  errors = %5d (out of %5d)  Ratio threshold = %6.4f  Sample threshold = %d\n\n", 
        nErrors, p->DataTY.nSamples * p->DataTY.nSampleBits, ChangeThresh, Threshold ); fflush( stdout );
    vImpls = Vec_IntAlloc( 1 << pLayer->LutSize );
    vGrads = Vec_IntAlloc( 1 << pLayer->LutSize );
    for ( Iter = 0; Iter < IterMax; Iter++ )
    {
        clock_t clkStart = clock();
        // initialize care info
        Lgn_NetForEachLayer( p, pLayer, i )
            Vec_WrdFill( &pLayer->vCare, pLayer->nLuts*p->nSimWords, i == Lgn_NetLayerNum(p)-1 ? ~((word)0) : 0 );
        // consider layers in the reverse order
        Lgn_NetForEachLayerReverse( p, pLayer, i )
        {
            int nLutsCare = 0, nLutsChanged = 0, nMintsCare = 0, nMintsChanged = 0;
            int nTotalMints = (1 << pLayer->LutSize)*pLayer->nLuts;
            Lgn_LayerForEachLut( pLayer, iLut )
            {
                int k, s, iNode, Impls, Mint, nVotes, fHaveCares = 0, fChanges = 0;
                word * pTruth  = Lgn_LayerLutTruth( pLayer, iLut );
                word * pSims   = Lgn_LayerLutSims( pLayer, iLut );
                word * pCare   = Lgn_LayerLutCare( pLayer, iLut );
                Lgn_Layer_t * pLayerPrev = Lgn_LayerPrev(pLayer);
                word * pFaninSims[16], * pFaninCare[16]; 
                Lgn_LutForEachFanin( pLayer, iLut, iNode, k )
                {
                    pFaninSims[k] = pLayerPrev ? Lgn_LayerLutSims(pLayerPrev, iNode) : Lgn_NetSimsIn(p, iNode);
                    pFaninCare[k] = pLayerPrev ? Lgn_LayerLutCare(pLayerPrev, iNode) : NULL;
                }
                // collect minterm statistics
                Vec_IntFill( vImpls, 1 << pLayer->LutSize, -1 );
                Vec_IntFill( vGrads, 1 << pLayer->LutSize,  0 );
                for ( s = 0; s < p->DataTY.nSamples; s++ )
                {
                    // skip don't-care data-samples
                    if ( !Abc_TtGetBit(pCare, s) )
                        continue;
                    fHaveCares = 1;
                    nLutsCare++;
                    // find fanin minterm
                    Mint = 0;
                    for ( k = 0; k < pLayer->LutSize; k++ )
                        if ( Abc_TtGetBit(pFaninSims[k], s) )
                            Mint |= 1 << k;
                    // find which inputs are implied
                    Impls = Vec_IntEntry( vImpls, Mint );
                    if ( Impls == -1 )
                    {
                        nMintsCare++;
                        Impls = Lgn_NetImplications( pTruth, pLayer->LutSize, Mint );
                        Vec_IntWriteEntry( vImpls, Mint, Impls );
                    }
                    // propagate cares to the fanins
                    if ( pLayerPrev )
                        for ( k = 0; k < pLayer->LutSize; k++ )
                            if ( (Impls >> k) & 1 )
                                Abc_TtSetBit(pFaninCare[k], s);
                    // add to the gradient
                    if ( Abc_TtGetBit(pSimErrs, s) ) // error
                        Vec_IntAddToEntry( vGrads, Mint,  1 ); // flip
                    else
                        Vec_IntAddToEntry( vGrads, Mint, -1 ); // no flip
                }
                if ( !fHaveCares )
                    continue;
                // skip minterms above the threshold
                Vec_IntForEachEntry( vGrads, nVotes, Mint )
                    if ( nVotes >= Threshold )
                    {
                        //printf( "LUT %3d changing mint %3d based on %3d votes.\n", iLut, Mint, nVotes );
                        Abc_TtXorBit( pTruth, Mint ), fChanges = 1, nMintsChanged++;
                    }
                nLutsChanged += fChanges;
                //printf( "LUT %2d ", iLut );
                //Vec_IntPrint( vGrads );
            }
            //printf( "Layer %d  LUTs  : Total = %6d.  Care = %6d (%6.2f %%).  Changed = %6d (%6.2f %%).\n", 
            printf( "Layer %d  LUTs  : Total = %6d.                             Changed = %6d (%6.2f %%).\n", 
                i, pLayer->nLuts, 
                //nLutsCare/p->DataTY.nSamples, 100.0*nLutsCare/p->DataTY.nSamples/pLayer->nLuts, 
                nLutsChanged,                 100.0*nLutsChanged/pLayer->nLuts ); 
            printf( "Layer %d  Mints : Total = %6d.  Care = %6d (%6.2f %%).  Changed = %6d (%6.2f %%).\n", 
                i, nTotalMints,  
                nMintsCare,    100.0*nMintsCare/nTotalMints,    
                nMintsChanged, 100.0*nMintsChanged/nTotalMints );
        }
        // resimulate 
        Lgn_NetForEachLayer( p, pLayer, i )
            Lgn_LayerForEachLut( pLayer, iLut )
                Lgn_LutSimulate( pLayer, iLut, NULL, NULL );
        // measure statistic
        nErrors = Lgn_NetCountErrors(p);
        printf( "Training iteration %2d:  ", Iter );
        printf( "Resulting errors = %5d  ", nErrors ); fflush( stdout );
        printf( "Time = %6.2f sec\n\n", (float)(clock() - clkStart)/CLOCKS_PER_SEC );
    }
    Vec_IntFree( vImpls );
    Vec_IntFree( vGrads );
    return Lgn_NetAccuracy( p, 0 );
}



/**Function*************************************************************

  Synopsis    [Training based on single cubes (exact).]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
static inline int Lgn_NetCubeDeriveOne( word * pOutSim, int nWords, word * pFinal, word ** pFanSims, int nFans, int Guide )
{
    int i, Count = 0;
    Abc_TtFill( pOutSim, nWords );
    for ( i = 0; i < nFans; i++ )
    {
        if ( Guide % 3 == 0 )      // pos lit
            Abc_TtAnd( pOutSim, pOutSim, pFanSims[i], nWords, 0 );
        else if ( Guide % 3 == 1 ) // neg lit
            Abc_TtSharp( pOutSim, pOutSim, pFanSims[i], nWords );
        Guide /= 3;
    }
    assert( Guide == 0 || Guide == 1 );
    if ( Guide == 1 )
        Abc_TtNot( pOutSim, nWords );
    for ( i = 0; i < nWords; i++ )
        Count += Abc_TtCountOnes( ~pOutSim[i] ^ pFinal[i] );
    return Count;
}
int Lgn_NetCubeDeriveExact( word * pOutSim, int nWords, word * pFinal, word ** pFanSims, int nFans, int * pWeightBest )
{
    int Weight, WeightBest = 0, GuideBest = -1;
    int i, Limit = 2;
    for ( i = 0; i < nFans; i++ )
        Limit *= 3;
    for ( i = 0; i < Limit; i++ )
    {
        int Weight = Lgn_NetCubeDeriveOne( pOutSim, nWords, pFinal, pFanSims, nFans, i );
        if ( WeightBest < Weight )
        {
            WeightBest = Weight;
            GuideBest = i;
        }
    }
    Weight = Lgn_NetCubeDeriveOne( pOutSim, nWords, pFinal, pFanSims, nFans, GuideBest );
    assert( Weight == WeightBest );
    *pWeightBest = WeightBest;
    return GuideBest;
}

/**Function*************************************************************

  Synopsis    [Training based on single cubes (heuristic).]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
static inline int Lgn_NetCubeDeriveGuide( int * pPolars, int nFans )
{
    int i, Guide = pPolars[nFans];
    assert( pPolars[nFans] == 0 || pPolars[nFans] == 1 );
    for ( i = nFans - 1; i >= 0; i-- )
    {
        assert( pPolars[i] >= 0 && pPolars[i] <= 2 );
        Guide = Guide * 3 + pPolars[i];
    }
    return Guide;
}
static inline int Lgn_NetCubeDeriveOneHeu( word * pOutSim, int nWords, word * pFinal, word ** pFanSims, int nFans, int * Polars )
{
    int i, Count = 0;
    Abc_TtFill( pOutSim, nWords );
    for ( i = 0; i < nFans; i++ )
    {
        if ( Polars[i] == 0 )      // pos lit
            Abc_TtAnd( pOutSim, pOutSim, pFanSims[i], nWords, 0 );
        else if ( Polars[i] == 1 ) // neg lit
            Abc_TtSharp( pOutSim, pOutSim, pFanSims[i], nWords );
    }
    assert( Polars[nFans] == 0 || Polars[nFans] == 1 );
    if ( Polars[nFans] == 1 )
        Abc_TtNot( pOutSim, nWords );
    for ( i = 0; i < nWords; i++ )
        Count += Abc_TtCountOnes( ~pOutSim[i] ^ pFinal[i] );
    return Count;
}
// returns 1 if another variable is successfully added
int Lgn_NetCubeDeriveHeuAddOne( word * pOutSim, int nWords, word * pFinal, word ** pFanSims, int nFans, int * Polars, int * pWeightBest )
{
    int i, n, Weight, WeightBest = 0, VarBest, PolarBest;
    for ( i = 0; i < nFans; i++ )
    {
        if ( Polars[i] != -1 )
            continue;
        for ( n = 0; n < 2; n++ )
        {
            assert( Polars[i] == -1 );
            Polars[i] = n;
            Weight    = Lgn_NetCubeDeriveOneHeu( pOutSim, nWords, pFinal, pFanSims, nFans, Polars );
            Polars[i] = -1;
            if ( WeightBest < Weight )
            {
                WeightBest = Weight;
                VarBest    = i;
                PolarBest  = n;
            }
        }
    }
    if ( *pWeightBest >= WeightBest )
        return 0;
    *pWeightBest    = WeightBest;
    Polars[VarBest] = PolarBest;
    return 1;
}
// returns best weight
int Lgn_NetCubeDeriveHeuPol( word * pOutSim, int nWords, word * pFinal, word ** pFanSims, int nFans, int * Polars )
{
    int i, WeightBest = 0;
    for ( i = 0; i < nFans; i++ )
        if ( !Lgn_NetCubeDeriveHeuAddOne(pOutSim, nWords, pFinal, pFanSims, nFans, Polars, &WeightBest) )
            break;
    for ( i = 0; i < 20; i++ )
        if ( Polars[i] == -1 )
            Polars[i] = 2;
    return WeightBest;
}
int Lgn_NetCubeDeriveHeu( word * pOutSim, int nWords, word * pFinal, word ** pFanSims, int nFans, int * pWeightBest )
{
    int i, * pPolarBest, Polars0[32], Polars1[32], Weight0, Weight1, Weight, GuideBest;
    assert( nFans < 20 );
    for ( i = 0; i < 20; i++ )
        Polars0[i] = Polars1[i] = -1;
    Polars0[nFans] = 0;
    Polars1[nFans] = 1;
    Weight0 = Lgn_NetCubeDeriveHeuPol( pOutSim, nWords, pFinal, pFanSims, nFans, Polars0 );
    Weight1 = Lgn_NetCubeDeriveHeuPol( pOutSim, nWords, pFinal, pFanSims, nFans, Polars1 );
    if ( Weight0 >= Weight1 )
    {
        pPolarBest   = Polars0;
        *pWeightBest = Weight0;
    }
    else
    {
        pPolarBest   = Polars1;
        *pWeightBest = Weight1;
    }
    GuideBest = Lgn_NetCubeDeriveGuide( pPolarBest, nFans );
    Weight = Lgn_NetCubeDeriveOne( pOutSim, nWords, pFinal, pFanSims, nFans, GuideBest );
    assert( Weight == *pWeightBest );
    return GuideBest;
}


/**Function*************************************************************

  Synopsis    [Training based on single cubes.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
int Lgn_CountLits( int Guide, int LutSize )
{
    int k, nLits = 0;
    for ( k = 0; k < LutSize; k++, Guide /= 3 )
        nLits += (Guide % 3) != 2;
    return nLits;
}
void Lgn_PrintPolars( Lgn_Layer_t * pLayer, int iLut, int Guide, int WeightBest )
{
    int k, pPolars[32], nLits = 0;
    printf( "Layer %2d.  LUT %5d.  Guide = %6d.  ", pLayer->Id, iLut, Guide );
    for ( k = 0; k < pLayer->LutSize; k++, Guide /= 3 )
        pPolars[k] = (char)(Guide % 3), nLits += (Guide % 3) != 2;
    assert( Guide == 0 || Guide == 1 );
    printf( "Lits = %2d.  Polars: ", nLits );
    for ( k = 0; k < pLayer->LutSize; k++ )
        printf( "%d", pPolars[k] );
    printf( ":%d.  Weight = %d.\n", Guide, WeightBest );
}
float Lgn_NetCubeTrain( Lgn_Net_t * p, char * pFileNameX, char * pFileNameY, int nRandom )
{
    float Result = 0;
    Lgn_Layer_t * pLayer; int i, iLut;
    if ( !Lgn_DataLoad(&p->DataTX, pFileNameX) )
        return 0;
    if ( !Lgn_DataLoad(&p->DataTY, pFileNameY) )
        return 0;
    if ( nRandom > 0 )
    {
        // randomize training labels
        word * pDataTemp;
        assert( p->DataTY.nSampleBits == 1 );
        //p->DataTY.pData = Lgn_PermuteLabels( pDataTemp = p->DataTY.pData, p->DataTY.nSamples );
        p->DataTY.pData = Lgn_PermuteLabelsPartial( pDataTemp = p->DataTY.pData, p->DataTY.nSamples, nRandom );
        ABC_FREE( pDataTemp ); 
        printf( "Randomly permuting labels with parameter nRandom = %d.\n", nRandom );
    }
    //assert( p->DataTY.nSampleBits == 1 );               // binary classification
    assert( p->DataTX.nSamples == p->DataTY.nSamples ); // data constistency
    Lgn_NetPrepare( p );
    Lgn_NetStartSimInfo( p, 0 );
    Lgn_NetForEachLayer( p, pLayer, i )
    {
        int nLits = 0;
        assert( pLayer->LutSize <= 32 );
        Lgn_LayerForEachLut( pLayer, iLut )
        {
            word * pSimsOut = Lgn_LayerLutSims( pLayer, iLut );
            //word * pFinal   = p->DataTY.pData;
            word * pFinal   = p->pOutValue[iLut % p->DataTY.nSampleBits];
            // collect fanins of this LUT
            word * pFaninSims[32];  int k, iFanin, Guide, WeightBest;
            Lgn_Layer_t * pLayerPrev = Lgn_LayerPrev(pLayer);
            Lgn_LutForEachFanin( pLayer, iLut, iFanin, k )
                pFaninSims[k] = i ? Lgn_LayerLutSims(pLayerPrev, iFanin) : Lgn_NetSimsIn(p, iFanin);
            // enumerate fanins in different polarities
            //Guide = Lgn_NetCubeDeriveExact( pSimsOut, p->nSimWords, pFinal, pFaninSims, pLayer->LutSize, &WeightBest );
            Guide = Lgn_NetCubeDeriveHeu( pSimsOut, p->nSimWords, pFinal, pFaninSims, pLayer->LutSize, &WeightBest );
            Vec_IntWriteEntry( &pLayer->vPolars, iLut, Guide );

            //Lgn_PrintPolars( pLayer, iLut, Guide, WeightBest );
            nLits += Lgn_CountLits( Guide, pLayer->LutSize );
        }
        //printf( "Finished training layer %2d (out of %2d)...\r", i, Lgn_NetLayerNum(p) );
#ifdef LGN_VERBOSE
        printf( "Finished training layer %2d (out of %2d)...  Average lit count = %4.1f\n", 
            i, Lgn_NetLayerNum(p), 1.0 * nLits / pLayer->nLuts );
        fflush( stdout );
#endif
    }
#ifdef LGN_VERBOSE
    printf( "Finished training %2d layers.                         \n", Lgn_NetLayerNum(p) );
    fflush( stdout );
#endif
    return Lgn_NetAccuracy( p, 0 );
}
float Lgn_NetCubeValidate( Lgn_Net_t * p, char * pFileNameX, char * pFileNameY )
{
    float Result = 0;
    Lgn_Layer_t * pLayer; int i, iLut;
    if ( !Lgn_DataLoad(&p->DataVX, pFileNameX) )
        return 0;
    if ( !Lgn_DataLoad(&p->DataVY, pFileNameY) )
        return 0;
    assert( p->DataTX.nSamples == p->DataTY.nSamples ); // data consistency
    Lgn_NetStartSimInfo( p, 1 );
    Lgn_NetForEachLayer( p, pLayer, i )
    {
        assert( pLayer->LutSize <= 32 );
        Lgn_LayerForEachLut( pLayer, iLut )
        {
            int    Guide    = Lgn_LayerLutPolar( pLayer, iLut );
            word * pSimsOut = Lgn_LayerLutSims( pLayer, iLut );
            //word * pFinal   = p->DataTY.pData;
            word * pFinal   = p->pOutValue[iLut % p->DataTY.nSampleBits];
            // collect fanins of this LUT
            word * pFaninSims[32];  int k, iFanin;
            Lgn_Layer_t * pLayerPrev = Lgn_LayerPrev(pLayer);
            Lgn_LutForEachFanin( pLayer, iLut, iFanin, k )
                pFaninSims[k] = i ? Lgn_LayerLutSims(pLayerPrev, iFanin) : Lgn_NetSimsIn(p, iFanin);
            // enumerate fanin products in different polarities
            Lgn_NetCubeDeriveOne( pSimsOut, p->nSimWords, pFinal, pFaninSims, pLayer->LutSize, Guide );
            //printf( "Layer %d.  LUT %2d.  Guide = %5d.                      \n", i, iLut, Guide );
        }
#ifdef LGN_VERBOSE
        printf( "Finished validating layer %2d (out of %2d)...\r", i, Lgn_NetLayerNum(p) );
        fflush( stdout );
#endif
    }
#ifdef LGN_VERBOSE
    printf( "Finished validating %2d layers.                         \n", Lgn_NetLayerNum(p) );
    fflush( stdout );
#endif
    return Lgn_NetAccuracy( p, 1 );
}

/**Function*************************************************************

  Synopsis    [Input file processing.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
int Lgn_NetCheckSpec( Vec_Int_t * vSpec )
{
    int i, nLuts, LutSize, Status = 1;
    if ( Vec_IntSize(vSpec) % 2 )
    {
        printf( "Expecting an even number of entries in the LogiNet spec file:\n" );
        printf( "{(layer0_lut_count, layer0_lut_size), (layer1_lut_count, layer1_lut_size), ... }\n" );
        Status = 0;
    }
    Vec_IntForEachEntryDouble( vSpec, nLuts, LutSize, i )
    {
        if ( nLuts < 1 || nLuts > 10000000 )
        {
            printf( "The LUT count in layer %d is not reasonable (%d).\n", i/2, nLuts );
            Status = 0;
        }
        if ( LutSize < 1 || LutSize > 16 )
        {
            printf( "The LUT size in layer %d is not reasonable (%d).\n", i/2, LutSize );
            Status = 0;
        }
    }
#ifdef LGN_VERBOSE
    printf( "LogicNet: " );
    Vec_IntForEachEntryDouble( vSpec, nLuts, LutSize, i )
        printf( "%d=(%d,%d) ", i/2, nLuts, LutSize );
    printf( "\n" );
#endif
    return Status;
}
Vec_Int_t * Lgn_NetReadNetSpec( char * pFileName )
{
    Vec_Int_t * vSpec;
    char * pToken, pBuffer[LGN_LINE_MAX];
    FILE * pFile = fopen( pFileName, "rb" );
    if ( pFile == NULL )
    {
        printf( "Failed to open file with LogicNet Spec \"%s\".\n", pFileName );
        return NULL;
    }
    vSpec = Vec_IntAlloc( 100 );
    while ( fgets( pBuffer, LGN_LINE_MAX, pFile ) )
    {
        assert( strlen(pBuffer) < LGN_LINE_MAX-1 );
        if ( pBuffer[0] == '#' || (pBuffer[0] == '/' && pBuffer[1] == '/') )
            continue;
        pToken = strtok( pBuffer, " \n\t\r" );
        while ( pToken )
        {
            Vec_IntPush( vSpec, atoi(pToken) );
            pToken = strtok( NULL, " \n\t\r" );
        }
    }
    fclose( pFile );
    return vSpec;
}
int Lgn_NetReadFileList( char * pFileName, char * pFileList[16] )
{
    int nFiles = 0;
    char * pToken, pBuffer[LGN_LINE_MAX];
    FILE * pFile = fopen( pFileName, "rb" );
    if ( pFile == NULL )
    {
        printf( "Failed to open file with input file list \"%s\".\n", pFileName );
        return 0;
    }
    while ( fgets( pBuffer, LGN_LINE_MAX, pFile ) )
    {
        assert( strlen(pBuffer) < LGN_LINE_MAX-1 );
        if ( pBuffer[0] == '#' || (pBuffer[0] == '/' && pBuffer[1] == '/') )
            continue;
        pToken = strtok( pBuffer, " \n\t\r" );
        while ( pToken && nFiles <= 16 )
        {
            pFileList[nFiles++] = Abc_UtilStrsav(pToken);
            pToken = strtok( NULL, " \n\t\r" );
        }
    }
    fclose( pFile );
    return nFiles;
}

/**Function*************************************************************

  Synopsis    [Top-level procedure.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
char * Lgn_NetFileName( char * pFileName, char * pExt )
{
    char * pSpot, * pRes = ABC_CALLOC( char, strlen(pFileName)+strlen(pExt)+10 );
    strcat( pRes, pFileName );
    pSpot = strrchr( pRes, '.' );
    if ( pSpot != NULL )
        *pSpot = '\0';
    strcat( pRes, pExt );
    return pRes;
}
char * Lgn_NetPrefName( char * pRes )
{
    char * pSpot = pRes, * pTemp;
    for ( pTemp = pRes; *pTemp; pTemp++ )
        if ( *pTemp == '\\' || *pTemp == '/' )
            pSpot = pTemp + 1;
    return pSpot;
}
void Lgn_NetRun( int argc, char ** argv, Vec_Int_t * vSpec, char * pFileList[4], int nRandom )
{
    char * pFileNameTX   = pFileList[0];
    char * pFileNameTY   = pFileList[1];
    char * pFileNameVX   = pFileList[2];
    char * pFileNameVY   = pFileList[3];
    char * pFileNameVer  = Lgn_NetFileName( pFileList[0], ".v"   );
    char * pFileNameLog  = Lgn_NetFileName( pFileList[0], ".log" );
    char * pFileNamePref = Lgn_NetFileName( pFileList[0], "_"    );
    FILE * pFileLog      = fopen( pFileNameLog, "wb" ); int i;

    Lgn_Net_t * pNet = Lgn_NetAlloc( vSpec );

    abctime clkT     = clock();
    //float ResultT    = Lgn_NetTrain( pNet, pFileNameTX, pFileNameTY, nRandom );
    //float ResultT    = Lgn_NetBackPropResimulate( pNet, pFileNameTX, pFileNameTY );
    //float ResultT    = Lgn_NetBackPropSensitize( pNet, pFileNameTX, pFileNameTY );
    float ResultT    = Lgn_NetCubeTrain( pNet, pFileNameTX, pFileNameTY, nRandom );
    abctime clkTT    = clock() - clkT;

    abctime clkV     = clock();
    //float ResultV    = Lgn_NetValidate( pNet, pFileNameVX, pFileNameVY );
    float ResultV    = Lgn_NetCubeValidate( pNet, pFileNameVX, pFileNameVY );
    abctime clkVV    = clock() - clkV;

#ifdef LGN_VERBOSE
    printf( "Accuracy on %6d training   samples = %8.2f %%.  ", pNet->DataTX.nSamples, ResultT );
    Abc_PrintTime( 1, "Time", clkTT );
    fflush( stdout );

    printf( "Accuracy on %6d validation samples = %8.2f %%.  ", pNet->DataVX.nSamples, ResultV );
    Abc_PrintTime( 1, "Time", clkVV );
    fflush( stdout );

    fprintf( pFileLog, "Command line was: " );
    for ( i = 0; i < argc; i++ )
        fprintf( pFileLog, " %s", argv[i] );
    fprintf( pFileLog, "\n" );
#endif

    fprintf( pFileLog, "Using training   data from files \"%s\" and \"%s\".\n", pFileNameTX, pFileNameTY );
    fprintf( pFileLog, "Using validation data from files \"%s\" and \"%s\".\n", pFileNameVX, pFileNameVY );

    fprintf( pFileLog, "Accuracy on %6d training   samples = %8.2f %%.  ", pNet->DataTX.nSamples, ResultT );
    fprintf( pFileLog, "Time = %9.2f sec\n", 1.0*((double)(clkTT))/((double)CLOCKS_PER_SEC) );
    fflush( pFileLog );

    fprintf( pFileLog, "Accuracy on %6d validation samples = %8.2f %%.  ", pNet->DataVX.nSamples, ResultV );
    fprintf( pFileLog, "Time = %9.2f sec\n", 1.0*((double)(clkVV))/((double)CLOCKS_PER_SEC) );
    fflush( pFileLog );

    //Lgn_NetDumpVerilog( pNet, pFileNameVer );
    //Lgn_NetDumpVerilogCube( pNet, pFileNameVer, "" );
    Lgn_NetDumpVerilogCube( pNet, pFileNameVer, Lgn_NetPrefName(pFileNamePref) );
#ifdef LGN_VERBOSE
    printf(            "Finished dumping file \"%s\".\n", pFileNameVer );
#endif
    fprintf( pFileLog, "Finished dumping file \"%s\".\n", pFileNameVer );

    fclose( pFileLog );
    ABC_FREE( pFileNameVer );
    ABC_FREE( pFileNameLog );
    ABC_FREE( pFileNamePref );
    Lgn_NetFree( pNet );
}
int main( int argc, char ** argv )
{
    int nRandom = 0;
    Vec_Int_t * vSpec = NULL;
    char * pNameFileList = NULL;
    char * pFileList[16] = {NULL};
    int i, nFiles = 0;
    //Lgn_NetMnistConvert();     return 1;
    //Lgn_NetCifar100Convert();  return 1;
    //Lgn_NetImplicationsTest(); return 1;
    if ( argc != 3 && argc != 6 )
    {
        char * pName1 = strrchr( argv[0], '\\' );
        char * pName2 = strrchr( argv[0], '/' );
        char * pName = pName1 ? pName1+1 : pName2 ? pName2+1 : argv[0];
/*
        printf( "usage: %s version %d is a LogicNet toolbox expecting two arguments\n", pName, LGN_VER_NUM );
        printf( "               <file_name_spec> is a text file describing LogicNet architecture\n" );
        printf( "               <file_name_list> is a text file listing four data files (tx, ty, vx, vy)\n" );
        printf( "           example: \"%s <file_name_spec> <file_name_list>\"\n", pName );
        printf( "                where <file_name_spec> is\n" );
        printf( "                    1024 10\n" );
        printf( "                    1024 10\n" );
        printf( "                    1024 10\n" );
        printf( "                    1024 10\n" );
        printf( "                    1024 10\n" );
*/
        printf( "usage: %s version %d is a LogicNet toolbox expecting exactly 6 arguments\n", pName, LGN_VER_NUM );
        printf( "               four space-separated integers describe LogicNet architecture\n" );
        printf( "               <file_name_list> is a text file listing four data files (tx, ty, vx, vy)\n" );
        printf( "           example: \"%s <depth> <width> <lutsize> <nouts> <file_name_list>\"\n", pName );
        printf( "                where \n" );
        printf( "                    <depth>   is the number of layers\n" );
        printf( "                    <width>   is the number of LUTs in each layer\n" );
        printf( "                    <lutsize> is the LUT size\n" );
        printf( "                    <nouts>   is the number of outputs\n" );

        printf( "                and <file_name_list> is\n" );
        printf( "                    data/mnist_60k_28_28_1_1.data\n" );
        printf( "                    data/mnist_60k_1.data        \n" );
        printf( "                    data/mnist_10k_28_28_1_1.data\n" );
        printf( "                    data/mnist_10k_1.data        \n" );
        fflush( stdout );
        return 1;
    }
    if ( argc == 3 )
    {
#ifdef LGN_VERBOSE
        printf( "Using LogicNet specification from \"%s\" and data files from \"%s\".\n", argv[1], argv[2] );
#endif
        vSpec = Lgn_NetReadNetSpec( argv[1] );
        pNameFileList = argv[2];
    }
    else if ( argc == 6 )
    {
        int Depth = atoi(argv[1]);
        int Width = atoi(argv[2]);
        int LSize = atoi(argv[3]);
        int nOuts = atoi(argv[4]);
        int i;
        vSpec = Vec_IntAlloc( 100 );
#ifdef LGN_VERBOSE
        printf( "Using LogicNet (D=%d, W=%d, K=%d, O=%d) and data files from \"%s\".\n", Depth, Width, LSize, nOuts, argv[5] );
#endif
        for ( i = 0; i < Depth; i++ )
            Vec_IntPushTwo( vSpec, i == Depth-1 ? nOuts : Width, LSize );
        //Vec_IntPrint( vSpec );
        pNameFileList = argv[5];
    }
    if ( Lgn_NetCheckSpec(vSpec) )
    {
        nFiles = Lgn_NetReadFileList( pNameFileList, pFileList );
        if ( nFiles == 4 )
            Lgn_NetRun( argc, argv, vSpec, pFileList, nRandom );
        else
            printf( "Expecting exactly four files in the file list: tx, ty, vx, vy.\n" );
    }
    // cleanup
    Vec_IntFreeP( &vSpec );
    for ( i = 0; i < nFiles; i++ )
        ABC_FREE( pFileList[i] );
    return 1;
}

////////////////////////////////////////////////////////////////////////
///                       END OF FILE                                ///
////////////////////////////////////////////////////////////////////////


ABC_NAMESPACE_IMPL_END

