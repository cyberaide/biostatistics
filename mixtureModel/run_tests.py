#!/usr/bin/env python

import os
import sys

def convert_vector_to_R(v):
    """
    Converts a python list to a string representation of the same vector in R
    """
    v = map(str,v)
    return "c("+",".join(v)+")"

def convert_matrix_to_R(m):
    """
    Converts a python list (or list of lists) to a string representation
    of the same matrix in R
    """
    vectors = [convert_vector_to_R(v) for v in m]
    return "rbind("+",".join(vectors)+")"

def generate_pdf(name,num,means,covs):
    """
    Generates a probability density function using the R statistical package
    and the MVTNORM library. 

    -name: the name of the output file
    -means: a list for the spectral mean of the distribution
    -covs: a covariance matrix for the data, must be square and dimension must match means
    """
    r_source = "library(mvtnorm)\n"
    r_source += "pdf = rmvnorm(%d, %s, %s)\n" % (num,convert_vector_to_R(means),convert_matrix_to_R(covs))
    r_source += 'write.table(pdf, file="%s", append=FALSE, quote=FALSE, sep=" ", row.names=FALSE, col.names=FALSE)\n' % name
    r_source += "q()\n"
    print r_source
    pstdin, pstdout = os.popen2("R --no-save")
    pstdin.write(r_source)
    pstdin.close()
    out = pstdout.read()
    print out
    
def combine_pdfs(pdfs):
    files = [open(f,"r") for f in pdfs]
    num_dimensions = len(files[0].next().split())
    files[0].seek(0)
    outf = open('combo.d%d.space' % num_dimensions,'w')
    for f in files:
        outf.write(f.read())
        f.close()
    outf.close()

def diag_matrix(n):
    m = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        m[i][i] = 1
    return m

# Create 7 2-D PDFs w/ 100,000 events each        
generate_pdf("pdf.d2a.space",10**5,[1,2],[[1,0.9],[0.9,1]])
generate_pdf("pdf.d2b.space",10**5,[3,4],[[1,0.8],[0.8,1]])
generate_pdf("pdf.d2c.space",10**5,[5,6],[[1,0.7],[0.7,1]])
generate_pdf("pdf.d2d.space",10**5,[7,8],[[1,0.6],[0.6,1]])
generate_pdf("pdf.d2e.space",10**5,[9,10],[[1,0.5],[0.5,1]])
generate_pdf("pdf.d2f.space",10**5,[11,12],[[1,0.4],[0.4,1]])
generate_pdf("pdf.d2g.space",10**5,[13,14],[[1,0.3],[0.3,1]])
combine_pdfs(["pdf.d2a.space","pdf.d2b.space","pdf.d2c.space","pdf.d2d.space","pdf.d2e.space","pdf.d2f.space","pdf.d2g.space"])

# Create 4 20-D PDFs w/ 25,000 events each
generate_pdf("pdf.d16a.space",25000,range(16),diag_matrix(16))
generate_pdf("pdf.d16b.space",25000,[2*i for i in range(16)],diag_matrix(16))
generate_pdf("pdf.d16c.space",25000,[-1*i for i in range(16)],diag_matrix(16))
generate_pdf("pdf.d16d.space",25000,[-2*i for i in range(16)],diag_matrix(16))
combine_pdfs(["pdf.d16a.space","pdf.d16b.space","pdf.d16c.space","pdf.d16d.space"])
