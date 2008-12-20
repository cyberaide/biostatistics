# Reads in an FCS and converts it to an ASCII file. Only the data segment is saved.
# Requires the rflowcyt package
#
# To install rflowcyt
# 	source("http://bioconductor.org/biocLite.R")
#	biocLite("rflowcyt")
# 
# Enter the following at the terminal to run the script
#
# R --no-save < convertFCS.r

if (!require(rflowcyt)) {
	stop("rflowcyt package needs to be installed.")
}

cmdArgs <- commandArgs();

fcs <- read.FCS(cmdArgs[3])
write.table(fcs@data, cmdArgs[4], sep=" ", row.names=FALSE, col.names=FALSE)
