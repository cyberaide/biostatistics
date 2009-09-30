#! /usr/bin/perl
$numArgs = $#ARGV + 1;
print "thanks, you gave me $numArgs command-line arguments.\n";
foreach $argnum (0 .. $#ARGV){

	printf "$ARGV[$argnum]\n";
}
# filename
my $fn = $ARGV[0];
if($fn =~ m/fcs$/)
{
	printf "FCS file must be converted";
	$newfn = $fn;
	#$newfn =~ s/fcs/txt/;
	$newfn = $newfn . ".dat";
	$rcommand = "cd $ARGV[13]/projects/cmeans/R/; R --no-save < convertFCS.r $fn $newfn; cd -";
	printf "$rcommand";
	system $rcommand;
	$fn = $newfn;
}

# dimentions
my $lcSTR = `wc -l $fn`;
my ($lc, $fn) = split(/ /, $lcSTR);
printf "Number of Events = $lc\n";
$NUM_EVENTS = $lc ;
# dimensions
open(MYFCSFILE, "$fn") || die("Could Not Open Provided File $fn");
my(@lines) = <MYFCSFILE>;
my(@tokens) = split(/[, \t]/,@lines[0]);
$NUM_DIMENSIONS = @tokens;
close(MYFCSFILE);
# clusters
my(@cluster_num_set) = split(/,/, $ARGV[1]);
# volume inclusion parameters
my(@volume_inc_set) = split(/,/, $ARGV[11]);



printf "Number of Dimensions = $NUM_DIMENSIONS\n";
my $NUM_THREADS = 384;
if($NUM_DIMENSIONS > 8){
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
	

for $CLUSTER_NUM  (@cluster_num_set) {
#    for $VOL_PARAM (@volume_inc_set) {
	if($CLUSTER_NUM > 50 || $NUM_EVENTS > 400000){
		$NUM_THREADS = $NUM_THREADS-32;
	}
	$command = "cd $ARGV[13]/projects/cmeans/; ./script.sh $CLUSTER_NUM $NUM_DIMENSIONS $NUM_EVENTS $ARGV[2] $ARGV[3] $ARGV[4] $ARGV[5] $ARGV[6] $ARGV[7] $ARGV[8] $ARGV[9] $ARGV[10] $ARGV[11]  $ARGV[12] $NUM_THREADS; cd -";
	printf "$command\n";
	system $command;
	$compile = "make clean -C $ARGV[13]/projects/cmeans/; make -C $ARGV[13]/projects/cmeans/";
	printf "$compile\n";    
	system $compile;
	$run= "$ARGV[13]/bin/linux/release/cmeans $fn";
	printf "$run\n";
	system $run;
#    }
}
