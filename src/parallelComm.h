#include <algorithm>
#include <cassert>
#include <map>
#include <unordered_map>
#include <vector>

#include "mpi.h"

namespace FVSAND {
  struct parallelComm
  {
  public:
    //
    // create the communication patterns based on a partitioning map obtained
    // from a graph partitioner, global numbering are expected to be 64 bit unsigned integers
    // while all local numbering are 32 bit integers
    // can work for both nodal partitioning or cell wise partitioning
    //
    void createCommPatterns( int myid,                                    // process id creating pattern
			     int *procmap,                                // processor id for each node (global )
			     int64_t *node2node,                          // node2node graph            (global) 
			     int  *ncon,                                  // number of connections per graph node (global)
			     uint64_t nnodes,                             // number of nodes (global)
			     int *nlocal,                                 // number of local owned nodes
			     int *nhalo,                                  // number of halo nodes internal interfaces only
			     std::vector<int> &node2nodelocal,            // local node connectivity including ghosts
			     std::vector<int> &ncon_local,                // number of connections per local graph node
			     std::vector<uint64_t> &local2global,         // local2global map linear
			     std::map <uint64_t, int > &global2local,     // global2local map (hashed)
			     std::map <int, std::vector<int>> &sndmap,    // map of send data (procid, local id of list of owned nodes)
			     std::map <int, std::vector<int>>  &rcvmap,   // map of recv data (procid, local id of ghost nodes);
			     MPI_Comm comm                                // MPI communicator
			     )
    {
      uint64_t *ngptr=new uint64_t [nnodes+1];
      std::vector <uint64_t> ptmp;
      std::map <int, std::vector<uint64_t> > pmap;
      std::vector<uint64_t> halo;

      int ierr=MPI_Comm_rank(comm,&myid);
      
      ngptr[0]=0;
      (*nlocal)=0;
      for(uint64_t i=0;i<nnodes;i++)
	{
	  if (procmap[i]==myid) {
	    local2global.push_back(i);
	    global2local.emplace(i,*nlocal);
	    (*nlocal)++;
	  }
	  if (i > 0) ngptr[i]=ngptr[i-1]+ncon[i-1];
	}
      ngptr[nnodes]=ngptr[nnodes-1]+ncon[nnodes-1];
      (*nhalo)=0;
      for(auto inode : local2global)
	{
	  ncon_local.push_back(ngptr[inode+1]-ngptr[inode]);      
	  for(int j=ngptr[inode];j<ngptr[inode+1];j++)
	    {
	      if (node2node[j] < 0) {
		node2nodelocal.push_back((int)node2node[j]); // local physical boundary condition
	      }
	      else if (node2node[j] > -1) {
		if (procmap[node2node[j]]!=myid) {
		  auto idxit=pmap.find(procmap[node2node[j]]);
		  if (idxit==pmap.end()) {	      
		    pmap[procmap[node2node[j]]]=std::vector<uint64_t>(1,inode);
		    sndmap[procmap[node2node[j]]]=std::vector<int>(1,global2local[inode]);
		  }
		  else {
		    // add the global id of ghost node and 
		    // the local node that connected to it into the 
		    // pmap,sndmap if it's not already included
		    if (std::count(pmap[procmap[node2node[j]]].begin(),
		     	            pmap[procmap[node2node[j]]].end(),
		     			    inode) == 0)
		    {
		    pmap[procmap[node2node[j]]].push_back(inode);
		    sndmap[procmap[node2node[j]]].push_back(global2local[inode]);
		    }
		  }
		  auto it = std::find(halo.begin(), halo.end(), node2node[j]);
		  if (it==halo.end()) {
		    int nodecount=(*nlocal)+(*nhalo);
		    halo.push_back(node2node[j]);        // global id of halo node
		    (*nhalo)++;
		    node2nodelocal.push_back(nodecount);      // connection is to a local ghost;
		  } else {
		    node2nodelocal.push_back(distance(halo.begin(),it)+(*nlocal));
		  }	    
		}
		else {	         
		  node2nodelocal.push_back(global2local[node2node[j]]); // connection is to a local node,
		  // add the local id of that here
		}
	      }
	    }
	}

      int b=(*nlocal);
      for (auto h : halo)
	{
	  local2global.push_back(h);
	  ncon_local.push_back(ngptr[h+1]-ngptr[h]);
	  for(int j=ngptr[h];j<ngptr[h+1];j++)
	    {
	      auto it = std::find(local2global.begin(), local2global.end(), node2node[j]);
	      if (it!=local2global.end() && distance(local2global.begin(),it) < (*nlocal)) {
	  	node2nodelocal.push_back(distance(local2global.begin(),it));
	      }
	      else {
	  	node2nodelocal.push_back(-1);
	      }
	    }
	  global2local.emplace(h,b);
	  b++;
	  //global2local[h.first]=h.second;
	}

      /*
      char fn[20];
      sprintf(fn,"c2c%d.dat",myid);
      FILE*fp=fopen(fn,"w");
      for(int i=0;i<(*nlocal+*nhalo);i++)
	{
	  for(int j=0;j<5;j++)
	    fprintf(fp,"%d ",node2nodelocal[5*i+j]);
	  fprintf(fp,"\n");
	}
      fclose(fp);
      */

      MPI_Request *ireq=new MPI_Request [pmap.size()*2];
      MPI_Status *istatus=new MPI_Status [pmap.size()*2];
      //
      // send global numbers of halo nodes to neighbor procs
      //
      uint64_t **buffer;
      buffer=(uint64_t **)malloc(sizeof(uint64_t *)*pmap.size());

      int k=0;
      for(auto p : pmap)
	{
	  auto procid=p.first;
	  auto sendlist=p.second;
	  buffer[k]=(uint64_t *)malloc(sizeof(uint64_t)*sendlist.size());	  
	  for(int b=0;b<sendlist.size();b++) buffer[k][b]=sendlist[b];
	  // why is this needed, why not use sendlist.data() ?
	  // MPI is unpredictable if I send sendlist.data() and
	  // gives garbage on the recieving end sometimes.
	  // I would think sendlist.data() is contiguous
	  MPI_Isend(buffer[k],sendlist.size(), MPI_UNSIGNED_LONG, 
	  	    procid, 0, comm, &(ireq[k]));
	  //MPI_Isend(sendlist.data(),sendlist.size(), MPI_UNSIGNED_LONG,
          //          procid, 0, comm, &(ireq[k]));
	  k++;
	}
      //
      // temporary map to receive the 64 bit global numberings
      //
      std::map <int, std::vector<uint64_t> > rmap;      
      int pcount=0;
      
      for(auto p : pmap)
	{
	  auto procid=p.first;
	  int ier=MPI_Probe(p.first,0,comm,&(istatus[pmap.size()+pcount]));
	  int datasize;
	  MPI_Get_count(&(istatus[pmap.size()+pcount]),MPI_LONG,&datasize);
	  rmap[procid]=std::vector<uint64_t>(datasize,0);
	  rcvmap[procid]=std::vector<int>(datasize,0);
	  MPI_Recv(rmap[procid].data(),rmap[procid].size(), MPI_LONG,
		   procid, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  pcount++;
	}
      MPI_Waitall(pmap.size(),ireq,istatus);
      //
      // wait for all send requests to complete
      //
      int err=MPI_Waitall(k,ireq,istatus);
      //
      // free the contiguous send buffers now
      //
      for(int k=0;k<pmap.size();k++) free(buffer[k]);
      free(buffer);
      //
      // convert the 64 bit global number to a 32 bit local number
      //
      for(auto &p :rcvmap)
	{
	  int procid=p.first;
	  for(int i=0;i<p.second.size();i++) {
	    p.second[i]=global2local[rmap[procid][i]];
	  }
	}

      delete [] ireq;
      delete [] istatus;
  
    }

    void exchangeDataDouble(double *q,
			    int nfields,
			    int ndof,
			    int istor,
			    const std::map <int, std::vector<int>> &sndmap,
			    const std::map <int, std::vector<int>> &rcvmap,
					std::unordered_map<int, std::vector<double> >& sndPacket,
      		std::unordered_map<int, std::vector<double> >& rcvPacket,
			    MPI_Comm comm)
    {
			const std::size_t num_requests = sndmap.size() + rcvmap.size();
      MPI_Request *ireq=new MPI_Request [ num_requests ];
      MPI_Status *istatus=new MPI_Status [ num_requests ];
      int myid;
      int ierr=MPI_Comm_rank(comm,&myid);
      int scale=(istor==0)?nfields:1;
      int stride=(istor==0)?1:ndof;

      int k=0;

			// post receives
			for(const auto& r : rcvmap)
			{
				const int fromRank = r.first;
				const int size     = r.second.size()*nfields;
				double* rbuffer		 = nullptr;
				if ( rcvPacket.find( fromRank ) == rcvPacket.end() )
				{
					rcvPacket[ fromRank ] = std::vector<double>( size ,0 );
					rbuffer = rcvPacket[ fromRank ].data();
				}
				else
				{
					rbuffer = rcvPacket[ fromRank ].data();
				}

				assert( rbuffer != nullptr );
				MPI_Irecv( rbuffer, size, MPI_DOUBLE, fromRank, 0, comm, &ireq[k++] );
			}

			// post sends
      for(const auto& s : sndmap)
			{
				const int toRank = s.first;
				const int size   = s.second.size() * nfields;
				double* sbuffer  = nullptr;
				
				if ( sndPacket.find(toRank) == sndPacket.end() )
				{
					sndPacket[ toRank ]=std::vector<double>( size );
					sbuffer = sndPacket[ toRank ].data();
				}
				else
				{
					sbuffer = sndPacket[ toRank ].data();
				}
				
				assert( sbuffer != nullptr );
				int m=0;
				for(auto v : s.second)
				{
					for(int i=0;i<nfields;i++)
						sbuffer[m++]=q[v*scale+i*stride];
				}

				MPI_Isend( sbuffer, size, MPI_DOUBLE, toRank, 0, comm, &ireq[k++] );
			}

      MPI_Waitall( num_requests ,ireq, istatus);

      double qnorm=0.0;
      //FILE *fp;
      //char fname[20];
      //sprintf(fname,"recv%d.dat",myid);
      //fp=fopen(fname,"w");
      for(const auto& r : rcvmap)
			{
				int m=0;
				const auto& rcvdata=rcvPacket[r.first];
				const auto& rcvlist=r.second;
				for(int k=0;k<rcvlist.size();k++)
				{
					int v=rcvlist[k];
					//fprintf(fp,"%d ",v);
					for(int i=0;i<nfields;i++)
					{
						double qv=rcvdata[m++];
						qnorm+=abs(qv-q[v*scale+i*stride]);
						//fprintf(fp,"%f ",q[v*scale+i*stride]);
						q[v*scale+i*stride]=qv;
					}
							//fprintf(fp,"\n");
				}

			}

      //fclose(fp);
      //printf("qnorm=%lf\n",qnorm);
      delete [] ireq;
      delete [] istatus;
    }
  };
}
