#This script will create a basic scatter plot of the generated clusters
#The medoids must be stored in medoids.dat and the data must be stored in output.dat
set terminal png nocrop enhanced size 1024,768
set output "chart.png"
set title "Generated Clusters"
set multiplot
set nokey
rgb(r,g,b) = int(r)*65536 + int(g)*256 + int(b)*256
splot "output.dat" using 1:2:3:(rgb($3,$3,$3)) with points lc rgb variable, \
	"medoids.dat" using 1:2:3:(rgb(0,0,0)) with points pt 7 ps 2 lc rgb variable
