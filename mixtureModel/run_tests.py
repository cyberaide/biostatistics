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
    
def combine_pdfs(pdfs,name=None):
    files = [open(f,"r") for f in pdfs]
    num_dimensions = len(files[0].next().split())
    files[0].seek(0)
    if not name:
        name = 'combo.d%d_%dcluster.space' % (num_dimensions,len(pdfs))
    outf = open(name,'w')
    for f in files:
        outf.write(f.read())
        f.close()
    outf.close()

def diag_matrix(n):
    m = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        m[i][i] = 1
    return m

def generate_testdata():
    # Create 10 16-D PDFs w/ 10,000 events each        
    generate_pdf("pdf.d16a.space",10**4,[i for i in range(16)],diag_matrix(16))
    generate_pdf("pdf.d16b.space",10**4,[2*i for i in range(16)],diag_matrix(16))
    generate_pdf("pdf.d16c.space",10**4,[-i for i in range(16)],diag_matrix(16))
    generate_pdf("pdf.d16d.space",10**4,[-2*i for i in range(16)],diag_matrix(16))
    generate_pdf("pdf.d16e.space",10**4,[3*i for i in range(16)],diag_matrix(16))
    generate_pdf("pdf.d16f.space",10**4,[-3*i for i in range(16)],diag_matrix(16))
    generate_pdf("pdf.d16g.space",10**4,[0.5*i for i in range(16)],diag_matrix(16))
    generate_pdf("pdf.d16h.space",10**4,[-0.5*i for i in range(16)],diag_matrix(16))
    generate_pdf("pdf.d16i.space",10**4,[5*i for i in range(16)],diag_matrix(16))
    generate_pdf("pdf.d16j.space",10**4,[-5*i for i in range(16)],diag_matrix(16))
    combine_pdfs(["pdf.d16a.space","pdf.d16b.space","pdf.d16c.space","pdf.d16d.space","pdf.d16e.space","pdf.d16f.space","pdf.d16g.space","pdf.d16h.space","pdf.d16i.space","pdf.d16j.space"])

    generate_pdf("pdf.d21a_100_event.space",100,range(21),diag_matrix(21))
    generate_pdf("pdf.d21b_100_event.space",100,list(reversed(range(21))),diag_matrix(21))
    combine_pdfs(["pdf.d21a_100_event.space","pdf.d21b_100_event.space"],"combo.d21_2cluster_200event.space")

    generate_pdf("pdf.d21a_1k_event.space",1000,range(21),diag_matrix(21))
    generate_pdf("pdf.d21b_1k_event.space",1000,list(reversed(range(21))),diag_matrix(21))
    combine_pdfs(["pdf.d21a_1k_event.space","pdf.d21b_1k_event.space"],"combo.d21_2cluster_2kevent.space")

    generate_pdf("pdf.d21a_10k_event.space",10000,range(21),diag_matrix(21))
    generate_pdf("pdf.d21b_10k_event.space",10000,list(reversed(range(21))),diag_matrix(21))
    combine_pdfs(["pdf.d21a_10k_event.space","pdf.d21b_10k_event.space"],"combo.d21_2cluster_20kevent.space")

    generate_pdf("pdf.d21a_50k_event.space",50000,range(21),diag_matrix(21))
    generate_pdf("pdf.d21b_50k_event.space",50000,list(reversed(range(21))),diag_matrix(21))
    combine_pdfs(["pdf.d21a_50k_event.space","pdf.d21b_50k_event.space"],"combo.d21_2cluster_100kevent.space")

    generate_pdf("pdf.d21a_100k_event.space",100000,range(21),diag_matrix(21))
    generate_pdf("pdf.d21b_100k_event.space",100000,list(reversed(range(21))),diag_matrix(21))
    combine_pdfs(["pdf.d21a_100k_event.space","pdf.d21b_100k_event.space"],"combo.d21_2cluster_200kevent.space")

    generate_pdf("pdf.d21a_250k_event.space",250000,range(21),diag_matrix(21))
    generate_pdf("pdf.d21b_250k_event.space",250000,list(reversed(range(21))),diag_matrix(21))
    combine_pdfs(["pdf.d21a_250k_event.space","pdf.d21b_250k_event.space"],"combo.d21_2cluster_500kevent.space")

    generate_pdf("pdf.d21a_500k_event.space",500000,range(21),diag_matrix(21))
    generate_pdf("pdf.d21b_500k_event.space",500000,list(reversed(range(21))),diag_matrix(21))
    combine_pdfs(["pdf.d21a_500k_event.space","pdf.d21b_500k_event.space"],"combo.d21_2cluster_1kkevent.space")

    generate_pdf("pdf.d21a_1kk_event.space",1000000,range(21),diag_matrix(21))
    generate_pdf("pdf.d21b_1kk_event.space",1000000,list(reversed(range(21))),diag_matrix(21))
    combine_pdfs(["pdf.d21a_1kk_event.space","pdf.d21b_1kk_event.space"],"combo.d21_2cluster_2kkevent.space")

    generate_pdf("pdf.d21a_2kk_event.space",2000000,range(21),diag_matrix(21))
    generate_pdf("pdf.d21b_2kk_event.space",2000000,list(reversed(range(21))),diag_matrix(21))
    combine_pdfs(["pdf.d21a_2kk_event.space","pdf.d21b_2kk_event.space"],"combo.d21_2cluster_4kkevent.space")

if __name__ == '__main__':
    generate_pdf("simple2d_cluster1.space",200,[2.0,2.0],[[1,0.1],[0.1,1.0]])
    generate_pdf("simple2d_cluster2.space",200,[-2.0,-2.0],[[1,-0.1],[-0.1,1.0]])
    generate_pdf("simple2d_cluster3.space",100,[5.5,2.0],[[1,0.2],[0.2,0.5]])
    combine_pdfs(["simple2d_cluster1.space","simple2d_cluster2.space","simple2d_cluster3.space"])
    #test_files = ['combo.d21_2cluster_2kevent.space','combo.d21_2cluster_20kevent.space','combo.d21_2cluster_100kevent.space','combo.d21_2cluster_200kevent.space','combo.d21_2cluster_500kevent.space','combo.d21_2cluster_1kkevent.space','combo.d21_2cluster_2kkevent.space']
    #import os
    #for test in test_files:
        #os.system('time ./emubin/gaussian 20 %s output' % test)
