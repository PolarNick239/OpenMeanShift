#include "../segm/msImageProcessor.h"

#include <cassert>

typedef double real_type;

void msImageProcessor::NewNonOptimizedFilter_omp(float sigmaS, float sigmaR,
                                                 float* msRawDataRes, std::queue<std::pair<size_t, size_t>>* workQueue, std::mutex* queueLock, std::vector<std::pair<size_t, size_t>>* workProcessed)
{
	std::queue<std::pair<size_t, size_t>> tmpQueue;
	std::vector<std::pair<size_t, size_t>> tmpWorkProcessed;
	std::mutex tmpMutex;

	if (msRawDataRes == nullptr && workQueue == nullptr && queueLock == nullptr && workProcessed == nullptr) {
		msRawDataRes = msRawData;
		workQueue = &tmpQueue;
		queueLock = &tmpMutex;
		workProcessed = &tmpWorkProcessed;
		tmpQueue.push(std::pair<int, int>(0, L));
	}

	//make sure that a lattice height and width have
	//been defined...
	if(!height)
	{
		ErrorHandler("msImageProcessor", "LFilter", "Lattice height and width are undefined.");
		return;
	}

	//re-assign bandwidths to sigmaS and sigmaR
	if(((h[0] = sigmaS) <= 0)||((h[1] = sigmaR) <= 0))
	{
		ErrorHandler("msImageProcessor", "Segment", "sigmaS and/or sigmaR is zero or negative.");
		return;
	}
	
	//define input data dimension with lattice
	int lN	= N + 2;
	
	// Traverse each data point applying mean shift
	// to each data point

   // let's use some temporary data
   real_type* sdata;
   sdata = new real_type[lN*L];

   // copy the scaled data
   int idxs, idxd;
   idxs = idxd = 0;
   if (N==3)
   {
      for(int i=0; i<L; i++)
      {
         sdata[idxs++] = (i%width)/sigmaS;
         sdata[idxs++] = (i/width)/sigmaS;
         sdata[idxs++] = data[idxd++]/sigmaR;
         sdata[idxs++] = data[idxd++]/sigmaR;
         sdata[idxs++] = data[idxd++]/sigmaR;
      }
   } else if (N==1)
   {
      for(int i=0; i<L; i++)
      {
         sdata[idxs++] = (i%width)/sigmaS;
         sdata[idxs++] = (i/width)/sigmaS;
         sdata[idxs++] = data[idxd++]/sigmaR;
      }
   } else
   {
      for(int i=0; i<L; i++)
      {
         sdata[idxs++] = (i%width)/sigmaS;
         sdata[idxs++] = (i%width)/sigmaS;
         for (int j=0; j<N; j++)
            sdata[idxs++] = data[idxd++]/sigmaR;
      }
   }
   // index the data in the 3d buckets (x, y, L)
   int* buckets;
   int* slist;
   slist = new int[L];
   int bucNeigh[27];

   real_type sMins; // just for L
   real_type sMaxs[3]; // for all
   sMaxs[0] = width/sigmaS;
   sMaxs[1] = height/sigmaS;
   sMins = sMaxs[2] = sdata[2];
   idxs = 2;
   real_type cval;
   for(int i=0; i<L; i++)
   {
      cval = sdata[idxs];
      if (cval < sMins)
         sMins = cval;
      else if (cval > sMaxs[2])
         sMaxs[2] = cval;

      idxs += lN;
   }

   int cBuck1, cBuck2, cBuck3, cBuck;
   const int nBuck1 = (int) (sMaxs[0] + 3);
   const int nBuck2 = (int) (sMaxs[1] + 3);
   const int nBuck3 = (int) (sMaxs[2] - sMins + 3);
   buckets = new int[nBuck1*nBuck2*nBuck3];
   for(int i=0; i<(nBuck1*nBuck2*nBuck3); i++)
      buckets[i] = -1;

   idxs = 0;
   for(int i=0; i<L; i++)
   {
      // find bucket for current data and add it to the list
      cBuck1 = (int) sdata[idxs] + 1;
      cBuck2 = (int) sdata[idxs+1] + 1;
      cBuck3 = (int) (sdata[idxs+2] - sMins) + 1;
      cBuck = cBuck1 + nBuck1*(cBuck2 + nBuck2*cBuck3);

      slist[i] = buckets[cBuck];
      buckets[cBuck] = i;

      idxs += lN;
   }
   // init bucNeigh
   idxd = 0;
   for (cBuck1=-1; cBuck1<=1; cBuck1++)
   {
      for (cBuck2=-1; cBuck2<=1; cBuck2++)
      {
         for (cBuck3=-1; cBuck3<=1; cBuck3++)
         {
            bucNeigh[idxd++] = cBuck1 + nBuck1*(cBuck2 + nBuck2*cBuck3);
         }
      }
   }
   real_type hiLTr = 80.0/sigmaR;
   // done indexing/hashing
	
	// proceed ...
#ifdef PROMPT
	msSys.Prompt("done.\nApplying mean shift (Using Lattice)... ");
#ifdef SHOW_PROGRESS
	msSys.Prompt("\n 0%%");
#endif
#endif

	while (true)
	{
		int workFrom;
		int workTo;
		{
			std::lock_guard<std::mutex> guard(*queueLock);
			if (workQueue->size() == 0) {
				break;
			}
			auto work = workQueue->front();
			workFrom = work.first;
			workTo = work.second;
			workProcessed->push_back(work);
			workQueue->pop();
		}
	#pragma omp parallel for schedule(dynamic, 4)
	for(int i = workFrom; i < workTo; i++)
	{
		int idxs, idxd;
		int j, k;
		int iterationCount;

		int cBuck1, cBuck2, cBuck3, cBuck;

		assert (lN <= 5);
		real_type yk[5];
		real_type Mh[5];

		real_type wsuml;
		real_type mvAbs, diff, el;
		real_type weight;

		// Assign window center (window centers are
		// initialized by createLattice to be the point
		// data[i])
      idxs = i*lN;
      for (j=0; j<lN; j++)
         yk[j] = sdata[idxs+j];
		
		// Calculate the mean shift vector using the lattice
		// LatticeMSVector(Mh, yk);
      /*****************************************************/
   	// Initialize mean shift vector
	   for(j = 0; j < lN; j++)
   		Mh[j] = 0;
   	wsuml = 0;
      // uniformLSearch(Mh, yk_ptr); // modify to new
      // find bucket of yk
      cBuck1 = (int) yk[0] + 1;
      cBuck2 = (int) yk[1] + 1;
      cBuck3 = (int) (yk[2] - sMins) + 1;
      cBuck = cBuck1 + nBuck1*(cBuck2 + nBuck2*cBuck3);
      for (j=0; j<27; j++)
      {
         idxd = buckets[cBuck+bucNeigh[j]];
         // list parse, crt point is cHeadList
         while (idxd>=0)
         {
            idxs = lN*idxd;
            // determine if inside search window
            el = sdata[idxs+0]-yk[0];
            diff = el*el;
            el = sdata[idxs+1]-yk[1];
            diff += el*el;

            if (diff < 1.0)
            {
               el = sdata[idxs+2]-yk[2];
               if (yk[2] > hiLTr)
                  diff = 4*el*el;
               else
                  diff = el*el;

               if (N>1)
               {
                  el = sdata[idxs+3]-yk[3];
                  diff += el*el;
                  el = sdata[idxs+4]-yk[4];
                  diff += el*el;
               }

               if (diff < 1.0)
               {
                  weight = 1-weightMap[idxd];
                  for (k=0; k<lN; k++)
                     Mh[k] += weight*sdata[idxs+k];
                  wsuml += weight;
               }
            }
            idxd = slist[idxd];
         }
      }
   	if (wsuml > 0)
   	{
		   for(j = 0; j < lN; j++)
   			Mh[j] = Mh[j]/wsuml - yk[j];
   	}
   	else
   	{
		   for(j = 0; j < lN; j++)
   			Mh[j] = 0;
   	}
      /*****************************************************/
		
		// Calculate its magnitude squared
		mvAbs = 0;
		for(j = 0; j < lN; j++)
			mvAbs += Mh[j]*Mh[j];
		
		// Keep shifting window center until the magnitude squared of the
		// mean shift vector calculated at the window center location is
		// under a specified threshold (Epsilon)
		
		// NOTE: iteration count is for speed up purposes only - it
		//       does not have any theoretical importance
		iterationCount = 1;
		while((mvAbs >= EPSILON)&&(iterationCount < LIMIT))
		{
			
			// Shift window location
			for(j = 0; j < lN; j++)
				yk[j] += Mh[j];
			
			// Calculate the mean shift vector at the new
			// window location using lattice
			// LatticeMSVector(Mh, yk);
         /*****************************************************/
         // Initialize mean shift vector
         for(j = 0; j < lN; j++)
            Mh[j] = 0;
         wsuml = 0;
         // uniformLSearch(Mh, yk_ptr); // modify to new
         // find bucket of yk
         cBuck1 = (int) yk[0] + 1;
         cBuck2 = (int) yk[1] + 1;
         cBuck3 = (int) (yk[2] - sMins) + 1;
         cBuck = cBuck1 + nBuck1*(cBuck2 + nBuck2*cBuck3);
         for (j=0; j<27; j++)
         {
            idxd = buckets[cBuck+bucNeigh[j]];
            // list parse, crt point is cHeadList
            while (idxd>=0)
            {
               idxs = lN*idxd;
               // determine if inside search window
               el = sdata[idxs+0]-yk[0];
               diff = el*el;
               el = sdata[idxs+1]-yk[1];
               diff += el*el;
               
               if (diff < 1.0)
               {
                  el = sdata[idxs+2]-yk[2];
                  if (yk[2] > hiLTr)
                     diff = 4*el*el;
                  else
                     diff = el*el;
                  
                  if (N>1)
                  {
                     el = sdata[idxs+3]-yk[3];
                     diff += el*el;
                     el = sdata[idxs+4]-yk[4];
                     diff += el*el;
                  }
                  
                  if (diff < 1.0)
                  {
                     weight = 1-weightMap[idxd];
                     for (k=0; k<lN; k++)
                        Mh[k] += weight*sdata[idxs+k];
                     wsuml += weight;
                  }
               }
               idxd = slist[idxd];
            }
         }
         if (wsuml > 0)
         {
            for(j = 0; j < lN; j++)
               Mh[j] = Mh[j]/wsuml - yk[j];
         }
         else
         {
            for(j = 0; j < lN; j++)
               Mh[j] = 0;
         }
         /*****************************************************/
			
			// Calculate its magnitude squared
			//mvAbs = 0;
			//for(j = 0; j < lN; j++)
			//	mvAbs += Mh[j]*Mh[j];
         mvAbs = (Mh[0]*Mh[0]+Mh[1]*Mh[1])*sigmaS*sigmaS;
         if (N==3)
            mvAbs += (Mh[2]*Mh[2]+Mh[3]*Mh[3]+Mh[4]*Mh[4])*sigmaR*sigmaR;
         else
            mvAbs += Mh[2]*Mh[2]*sigmaR*sigmaR;

			// Increment interation count
			iterationCount++;
		}

		// Shift window location
		for(j = 0; j < lN; j++)
			yk[j] += Mh[j];
		
		//store result into msRawData...
		for(j = 0; j < N; j++)
			msRawDataRes[N*i+j] = (float)(yk[j+2]*sigmaR);

		// Prompt user on progress
#ifdef SHOW_PROGRESS
		percent_complete = (float)(i/(float)(L))*100;
		msSys.Prompt("\r%2d%%", (int)(percent_complete + 0.5));
#endif
	
#ifdef MSSYS_PROGRESS
		// Check to see if the algorithm has been halted
		if((i%PROGRESS_RATE == 0)&&((ErrorStatus = msSys.Progress((float)(i/(float)(L))*(float)(0.8)))) == EL_HALT)
			break;
#endif
	}
	}
	
	// Prompt user that filtering is completed
#ifdef PROMPT
#ifdef SHOW_PROGRESS
	msSys.Prompt("\r");
#endif
	msSys.Prompt("done.");
#endif
	
	// de-allocate memory
   delete [] buckets;
   delete [] slist;
   delete [] sdata;

	// done.
	return;

}
