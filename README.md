# runrest

A wrapper for running [REST](https://github.com/cuhk-eda/REST), a reinforcement learning framework for constructing rectilinear Steiner Minimum tree (RSMT)

## Installation

```bash
git clone
cd runrest
pip install .
```

## How to use

```bash
runrest ./examples/input01.txt
```

### Example input file

x, y coordinates of the terminals are listed in the input file.
New line is used to seperate the nets.

`input01.txt:`

```text
5 0 0 2 4 3 1 4 6 5
```

### Result

The result will be printed to the console.  
Or you can use the '--output' option to save the result to a file.  

```text
3 1 3 2 0 2 0 4
```

It means the RSMT is constructed by the following edges:  
`(3, 1), (3, 2), (0, 2), (0, 4)`

Plot of the output RSMT:

![Result](./doc/example01_result.png)

## License

READ THIS LICENSE AGREEMENT CAREFULLY BEFORE USING THIS PRODUCT. BY USING THIS 
PRODUCT YOU INDICATE YOUR ACCEPTANCE OF THE TERMS OF THE FOLLOWING AGREEMENT. 
THESE TERMS APPLY TO YOU AND ANY SUBSEQUENT LICENSEE OF THIS PRODUCT.

License Agreement for REST

Copyright (c) 2022, The Chinese University of Hong Kong
All rights reserved.

CU-SD LICENSE (adapted from the original BSD license) Redistribution of the any 
code, with or without modification, are permitted provided that the conditions 
below are met. 

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name nor trademark of the copyright holder or the author may be 
   used to endorse or promote products derived from this software without 
   specific prior written permission.

4. Users are entirely responsible, to the exclusion of the author, for 
   compliance with (a) regulations set by owners or administrators of employed 
   equipment, (b) licensing terms of any other software, and (c) local, 
   national, and international regulations regarding use, including those 
   regarding import, export, and use of encryption software.

THIS FREE SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED 
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
SHALL THE AUTHOR OR ANY CONTRIBUTOR BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
LIMITED TO, EFFECTS OF UNAUTHORIZED OR MALICIOUS NETWORK ACCESS; PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
