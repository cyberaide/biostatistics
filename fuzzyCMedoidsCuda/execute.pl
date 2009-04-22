#! /usr/bin/perl
$totalArgs = 3;

$numArgs = $#ARGV + 1;
print "Received $numArgs command-line arguments.\n\n";

if ($numArgs < $totalArgs || $numArgs > $totalArgs) {
	die("Invalid number of arguments.\n")
}

#foreach $argnum (0 .. $#ARGV){
#        printf "$ARGV[$argnum]\n";
#}

# filename
my $fn = $ARGV[1];
if($fn =~ m/fcs$/)
{
        printf "FCS file must be converted";
        $newfn = $fn;
        #$newfn =~ s/fcs/txt/;
        $newfn = $newfn . ".dat";
        $rcommand = "cd ../R/; R --no-save < convertFCS.r $fn $newfn; cd -";
        printf "$rcommand";
        system $rcommand;
        $fn = $newfn;
}

# data points
my $lcSTR = `wc -l $fn`;
my ($lc, $fn) = split(/ /, $lcSTR);
printf "Number of Data Points = $lc\n";
$NUM_DATA_POINTS = $lc;

# dimensions
open(MYFCSFILE, "$fn") || die("Could Not Open Provided File $fn");
my(@lines) = <MYFCSFILE>;
my(@tokens) = split / /,@lines[0];
$NUM_DIMENSIONS = @tokens;
printf "Number of Dimensions = $NUM_DIMENSIONS\n";
close(MYFCSFILE);

# num clusters
$NUM_CLUSTERS = $ARGV[0];
printf "Number of Clusters = $NUM_CLUSTERS\n";

# determine number of threads
#my $NUM_THREADS = 384;
$NUM_THREADS = 384;
#$NUM_THREADS = int(sqrt($NUM_DATA_POINTS));

#if ($NUM_THREADS > 512) {
#	$NUM_THREADS = 512;
#}

if($NUM_DIMENSIONS > 8) {
	if($NUM_DIMENSIONS <= 10){
		$NUM_THREADS = 320;
	}
	elsif($NUM_DIMENSIONS <= 12){
		$NUM_THREADS = 256;
	}
	elsif($NUM_DIMENSIONS <= 16){
		$NUM_THREADS = 192;
	}
	elsif($NUM_DIMENSIONS <= 24){
		$NUM_THREADS = 128;
	}
	else{
		$NUM_THREADS = 64;
	}
}

printf "Number of Threads = $NUM_THREADS\n";
printf "Number of Blocks = $NUM_CLUSTERS\n";

# step size
$STEP_SIZE = int($NUM_DATA_POINTS / ($NUM_CLUSTERS * $NUM_THREADS));
$STEP_SIZE_MEMB = int($NUM_DATA_POINTS / ($NUM_CLUSTERS * 50));
printf "Thread Step Size = $STEP_SIZE\n";

# distance measure
$DIST_MEASURE = $ARGV[2];
printf "Distance Measure = $DIST_MEASURE\n";

# Create header
printf "\nCreating header file.\n\n";
system "./createHeader.sh $NUM_CLUSTERS $NUM_THREADS $NUM_DATA_POINTS $NUM_DIMENSIONS $STEP_SIZE $STEP_SIZE_MEMB $DIST_MEASURE";

# run make
printf "Cleaning and building the project.\n";
system "make clean";
system "make";
