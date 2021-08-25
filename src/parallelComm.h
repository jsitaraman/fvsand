#include<vector>
#include<map>
#include<algorithm>
#include "mpi.h"

namespace fvSand {
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
  std::map <uint64_t, int> halomap;
    
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
	      pmap[procmap[node2node[j]]]=std::vector<uint64_t>(1,node2node[j]);
	      sndmap[procmap[node2node[j]]]=std::vector<int>(1,global2local[inode]);
	    }
	    else {
             // add the global id of ghost node and 
	     // the local node that connected to it into the 
	     // pmap,sndmap if it's not already included
	     //if (std::count(pmap[procmap[node2node[j]]].begin(),
	     //	            pmap[procmap[node2node[j]]].end(),
              // 			    node2node[j]) == 0) {
	        pmap[procmap[node2node[j]]].push_back(node2node[j]);
	        sndmap[procmap[node2node[j]]].push_back(global2local[inode]);
	      //}
	    }
	    auto gnode=halomap.find(node2node[j]);
	    if (gnode==halomap.end()) {
	      int nodecount=(*nlocal)+(*nhalo);
	      node2nodelocal.push_back(nodecount);      // these are interface halo nodes
	      halomap[node2node[j]]=nodecount;          // map of global id to halo index
	      (*nhalo)++;
	      node2nodelocal.push_back(nodecount);      // connection is to a local ghost;
	    } else {
	      node2nodelocal.push_back(gnode->second);
	    }	    
	  }
	  else {	         
	    node2nodelocal.push_back(global2local[node2node[j]]); // connection is to a local node,
	                                                          // add the local id of that here
	  }
	}
      }
    }

  int b=0;
  for (auto h : halomap)
    {
      local2global.push_back(h.first);
      global2local[h.first]=h.second;
      b++;
    }
  
  MPI_Request *ireq=new MPI_Request [pmap.size()*2];
  MPI_Status *istatus=new MPI_Status [pmap.size()*2];
  //
  // send global numbers of halo nodes to neighbor procs
  //
  int k=0;
  for(auto p : pmap)
    {
      auto procid=p.first;
      auto sendlist=p.second;
      MPI_Isend(sendlist.data(),sendlist.size(), MPI_LONG, 
		procid, 0, MPI_COMM_WORLD, &ireq[k++]);
    }
  //
  // temporary map to receive the 64 bit global numberings
  //
  std::map <int, std::vector<uint64_t> > rmap;
  int pcount=0;
  for(auto p : pmap)
    {
      auto procid=p.first;
      //auto sendlist=p.second;
      int ier=MPI_Probe(p.first,0,comm,&(istatus[pmap.size()+pcount]));
      int datasize;
      MPI_Get_count(&(istatus[pmap.size()+pcount]),MPI_LONG,&datasize);
      rmap[procid]=std::vector<uint64_t>(datasize,0);
      rcvmap[procid]=std::vector<int>(datasize,0);
      MPI_Recv(rmap[procid].data(),rmap[procid].size(), MPI_LONG, 
		procid, 0, MPI_COMM_WORLD, &(istatus[pmap.size()+pcount]));
      pcount++;
   }
  MPI_Waitall(pmap.size(),ireq,istatus);
  //
  // convert the 64 bit global number to a 32 bit local number
  //
  for(auto p : rcvmap)
    {
      auto procid=p.first;
      auto rcvlist=p.second;
      for(int i=0;i<rcvlist.size();i++) rcvlist[i]=global2local[rmap[procid][i]];
    }

  delete [] ireq;
  delete [] istatus;
  
  }
 }  ;
}
