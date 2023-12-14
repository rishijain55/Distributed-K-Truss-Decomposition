# K-Truss Decomposition using Open MPI and OpenMP
## Overview
This project implements a distributed k-truss decomposition algorithm for undirected, unweighted, and simple graphs using Open MPI and OpenMP. K-truss decomposition is a graph analysis technique that identifies cohesive substructures within a graph.

## Dataset Description
The input graph G = (V, E) is represented as an undirected, unweighted, and simple graph with no self-loops. The input files are in binary format, with the following structure:

<4-bytes-representing n> <4-bytes-representing m>

«node-info-node-0»  
«node-info-node-1»  
...  
«node-info-node-(n-1)»  

Each node information block («node-info-node-i») includes:

<4-byte-representing node-i>  
<4-byte-representing-deg(i)>  
<neighbors>: <4-bytes-representing neighbour-1> ... <4-bytes-representing neighbour-deg(i)>  
Vertices are 0-based, and a corresponding metadata file (header.dat) contains the start offset for each node in the binary file.

## Building and Running
To build the program, execute the following command:

```
make
```
To run the program, use the provided PBS script:

```
Copy code
#!/bin/bash
# PBS ...
mpirun -np <no-of-processes> ./a3 --taskid=<1/2> \
--inputpath=<path-to-graph-input-file> \
--headerpath=<path-to-header-file> \
--outputpath=<path-to-output-file> \
--verbose=<0/1> \
--startk=<start-of-range> \
--endk=<end-of-range> \
```

> Note: For each k, we calculated k+2 truss.

## Task Execution Instructions

### Common Parameters
- **task_id (integer):** An integer (1, 2) based on the tasks described below. Defaults to 1.

- **inputpath (string):** The complete path to the input file.

- **headerpath (string):** The complete path to the header file.

- **outputpath (string):** The complete path where the output file should be saved. The output file is a normal text file.

- **verbose (binary flag):** A binary flag that takes values <0/1>. Defaults to 0. The output format changes based on this value.

### Output Format

#### Task 1
- The output file should contain results for each `k` in the range `[startk, endk]`, separated by a single space.
- Input to this task will contain the command line arguments `taskid`, `inputpath`, `headerpath`, `outputpath`, `verbose`, `startk`, and `endk`.
- For each `k` in the range `[startk, endk]`, the output file should contain a boolean value (0/1) indicating whether a group of size `k` exists.
- If the verbose flag is 1, and groups of size `k` exist, then:
  - On the new line (after 1), output the count of the `k`-size group(s) (say `c`).
  - In the next `c` subsequent lines, for each group, output the vertices which should be space-separated in any order.
  - These `c` groups can be in any order.

#### Task 2
- The output file should contain each influencer vertex separated with a single space.
- The social group is any group where each pair of friends is supported by `endk` friends.
- If there are no influencer vertices, the output should contain a single integer, i.e., `-1`.
- If the verbose flag is 1, then in addition to the influencer vertex:
  - The next line of the file should contain the count of the social groups (say `c`).
  - In the next `c` subsequent lines, list the members (or vertices) of these groups, which should be space-separated.
  - All the `c` social groups can be in any order, and within the group, the order doesn’t matter.


## Task-1 Approach

For Task 1, the following approach was followed:

1. We first distributed the graph into `p` parts - each processor owns some of the `n` nodes. For each edge `(u, v)`, we distribute it to some processor `p` that either is the owner of `u` or is the owner of `v`. We decide this using a relation on both nodes (Assign it to the node with less degree).

2. After distributing the nodes, we calculate the support of each edge in the `suppCalc` function.

3. Then we use the `minTruss` algorithm (just like assignment 2), which iteratively deletes the edges with support value that is less than `k`. This `minTruss` algorithm is implemented in our code in the `propLight()` function.

4. After these steps, we have final nodes on different processors, but we need connected components. So, we run a distributed BFS algorithm. We wrote this algorithm ourselves, by maintaining a frontier, and assigning colors to the same connected component. (This is needed only for `verbose = 1`)

5. Finally, we write the output to the file.

## Task-2 Approach

For Task 2, the following steps were taken:

1. We first found out the connected components of the graph using Distributed BFS after the process done in Task 1 with `verbose = 0`. After this step, all the nodes have the connected components of the graph. This is done by sending the connected components calculated by each node to all the other nodes using MPI Allgatherv.

2. We iterate over all the nodes and find out the nodes which are connected to more than `p` components. This process is done in a distributed manner. Each processor iterates over its own nodes and finds the nodes connected to more than `p` components.

3. To find out the nodes which are connected to more than `p` components, we use `pragma omp parallel for` to do the task in parallel. We use a critical section to avoid data race.

4. After this step, each processor has a list of nodes that are connected to more than `p` components. After this step, we use MPI Allgatherv to send this list to processor 0. After this step, processor 0 has the list of all the nodes which are connected to more than `p` components.

5. Processor 0 then writes this list to the file.
