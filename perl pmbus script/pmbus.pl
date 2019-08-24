#!/usr/bin/perl
use strict;

use bigint qw/hex/;


my       $Debug_Port=   0;
my       $Access_Port=  1;
my       $DP     =      0;
my       $AP   =        1;


my       $DP_IDCODE =   0;
my       $DP_CTRL_STAT= 1;
my       $DP_RDBUF  =   3;


my       $DP_ABORT    = 0;
my       $DP_SELECT  =  2;


my       $AP_CSW    =   0;
my       $AP_TAR   =    1;
my       $AP_DRW  =     3;


my       $OTP_BASE  =            0x12000000;
my       $OTPC_MODE =            0x12002000 ;
my       $OTPC_PTIME=            0x12002004 ;
my       $OTPC_RTIME =           0x12002008 ;
my       $OTPC_STS   =           0x12002010 ; 
my       $ADIM_BASE   =        0xD0000000;
my       $ADI_SLAVE_CTRL = 0xD0000000 ;
my       $ADI_RDWR_CTRL =        0xD0000010 ;

my @final;
open(FILE," pmbus.txt") or die("file not exists$!");
print("Enter label name\n");
my $label = <STDIN>;
chomp $label;
my $head = "header";
my $dat = "data";
print("Enter output file name\n");
my $filename=<STDIN>;
chomp($filename);

while(<FILE>) {

my($line) = $_;
chomp($line);
my @result = grep (/pmbus_write/, $line);
push (@final, @result); 
}

my $count=0;
my $countnum=1;
my $pmbus=1;
use Excel::Writer::XLSX;
			 
my $workbook  = Excel::Writer::XLSX->new( $filename );
my $worksheet = $workbook->add_worksheet();
	$worksheet->write( 0, 0, "Step" );
	$worksheet->write( 0, 1, "Label" );
	$worksheet->write(0, 2, "Capture" );
	$worksheet->write( 0, 3, "TMU Arm" );
	$worksheet->write( 0, 4, "MCU_Trigger" );
	$worksheet->write( 0, 5, "Instructions" );
	$worksheet->write( 0, 6, "In_OUT" );
	$worksheet->write( 0, 7, "CLK" );
	$worksheet->write( 0, 8, "In_OUT" );
	$worksheet->write( 0, 9, "DATA" );
	$worksheet->write( 0, 10, "Comments" );
	for(my $a=1;$a<=368;$a++)
		{
		$worksheet->write( $a, 0, "$a" );
		}
my @header;
my @data1;
my @data2;
my @data3;
my $value1;
my $lineno=1;
foreach my $value (@final)
	{
	$value=~ s/pmbus_write//;
	$value =~ s/;.*//;
	$value =~ s/0x//g;
	$value =~ s/,//g;
	$value=~ s/pmbaddr//;
	$value =~ s/[^[:alnum:]_-]//g;
	
	
	my @temphex= split(//, $value);
	
	$data1[0]=hex($temphex[0]);
	$data1[1]=hex($temphex[1]);
	$data2[0]=hex($temphex[2]);
	$data2[1]=hex($temphex[3]);
	$data3[0]=hex($temphex[4]);
	$data3[1]=hex($temphex[5]);
	
	


	
	$worksheet->write( $count+1, 1, "$label$pmbus" );
	$worksheet->write( $count+4, 1, "w2_addrs_$label$pmbus" );
	$worksheet->write( $count+13, 1, "w2_cmd_$label$pmbus" );
	$worksheet->write( $count+22, 1, "w2_data1_$label$pmbus" );
	$worksheet->write( $count+31, 1, "w2_data2_$label$pmbus" );
	$worksheet->write( $count+42, 1, "QMS_$label$pmbus" );
	$worksheet->write( $count+43, 4, "PSQGroup1" );
	
	
	$worksheet->write( $count+40, 5, "Burst: 2" );
	$worksheet->write( $count+41, 5, "Set Loop: 100" );
	$worksheet->write( $count+44, 5, "End Loop: QMS_$pmbus" );
	
	
	$worksheet->write( $count+1, 6, "1" );
	$worksheet->write( $count+2, 6, "1" );
	$worksheet->write( $count+3, 6, "1" );
	for(my $b=4;$b<=40;$b++)
		{
		$worksheet->write( $count+$b, 6, "0" );
		}
	$worksheet->write( $count+41, 6, "1" );
	$worksheet->write( $count+42, 6, "1" );
	$worksheet->write( $count+43, 6, "1" );
	$worksheet->write( $count+44, 6, "1" );
	$worksheet->write( $count+45, 6, "1" );
	
	
	$worksheet->write( $count+1, 7, "CL1_TS" );
	$worksheet->write( $count+2, 7, "CL1_TS" );
	$worksheet->write( $count+3, 7, "CL1_TS" );
	for(my $c=4;$c<=45;$c++)
		{
		$worksheet->write( $count+$c, 7, "CL2_TS" );
		}
		
	
	
	for(my $d=1;$d<=3;$d++)
		{
		$worksheet->write( $count+$d, 9, "DA1_TS" );
		}
	for(my $d=4;$d<=45;$d++)
	{
	$worksheet->write( $count+$d, 9, "DA2_TS" );
	}
	
	
	
		my $paritysum=0;
		my $paritybit;
		
		
		my @bitsData1;
		my @bitsData2;
		my @bitsData3;
		for (my $u=0; $u<=1;$u++)
		{
		my $bindata=sprintf("%04b", $data1[$u]);
		my @bitsData11 = split(//, $bindata);
		push(@bitsData1,@bitsData11);
		}
		
		for (my $u=0; $u<=1;$u++)
		{
		my $bindata=sprintf("%04b", $data2[$u]);
		my @bitsData22 = split(//, $bindata);
		push(@bitsData2,@bitsData22);
		}
		
		for (my $u=0; $u<=1;$u++)
		{
		my $bindata=sprintf("%04b", $data3[$u]);
		my @bitsData33 = split(//, $bindata);
		push(@bitsData3,@bitsData33);
		}
		
	
	
	
	
	$worksheet->write( $count+1, 8, "1" );
	$worksheet->write( $count+2, 8, "1" );
	for (my $g=3; $g<=11; $g++)
		{
		$worksheet->write( $count+$g, 8, "0" );
		}
	$worksheet->write( $count+12, 8, "1" );
	my $m1=0;
	for (my $g=13; $g<=20; $g++)
		{
		$worksheet->write( $count+$g, 8, "$bitsData1[$m1]" );
		$m1++;
		}
	$worksheet->write( $count+21, 8, "1" );
	
	my $m2=0;
	for (my $h=22; $h<=29;$h++)
		{
		$worksheet->write( $count+$h, 8, "$bitsData3[$m2]" );
		$m2++;
		}
	$worksheet->write( $count+30, 8, "1" );
	my $m3=0;
	for (my $h=31; $h<=38;$h++)
		{
		
		$worksheet->write( $count+$h, 8, "$bitsData2[$m3]" );
		$m3++;
		}
	$worksheet->write( $count+39, 8, "1" );
	$worksheet->write( $count+40, 8, "0" );
	$worksheet->write( $count+41, 8, "1" );
	$worksheet->write( $count+42, 8, "1" );
	$worksheet->write( $count+43, 8, "1" );
	$worksheet->write( $count+44, 8, "1" );
	$worksheet->write( $count+45, 8, "1" );
	
	$worksheet->write( $count+1, 10, "SCL=H & SDA=H" );
	$worksheet->write( $count+3, 10, "start" );
	$worksheet->write( $count+4, 10, "ADDRESS 1 MSB Upper Byte" );
	$worksheet->write( $count+5, 10, "ADDRESS 2" );
	$worksheet->write( $count+6, 10, "ADDRESS 3" );
	$worksheet->write( $count+7, 10, "ADDRESS 4 MSB Lower Byte" );
	$worksheet->write( $count+8, 10, "ADDRESS 5" );
	$worksheet->write( $count+9, 10, "ADDRESS 6" );
	$worksheet->write( $count+10, 10, "ADDRESS 7" );
	$worksheet->write( $count+11, 10, "WR=0 8" );
	$worksheet->write( $count+12, 10, "ACK 9" );
	$worksheet->write( $count+13, 10, "S7 1" );
	$worksheet->write( $count+14, 10, "S6 2" );
	$worksheet->write( $count+15, 10, "S5 3" );
	$worksheet->write( $count+16, 10, "S4 4" );
	$worksheet->write( $count+17, 10, "S3 5" );
	$worksheet->write( $count+18, 10, "S2 6" );
	$worksheet->write( $count+19, 10, "S1 7" );
	$worksheet->write( $count+20, 10, "S0 8" );
	$worksheet->write( $count+21, 10, "ACK 9" );
	
	$worksheet->write( $count+22, 10, "D7 1" );
	$worksheet->write( $count+23, 10, "D6 2" );
	$worksheet->write( $count+24, 10, "D5 3" );
	$worksheet->write( $count+25, 10, "D4 4" );
	$worksheet->write( $count+26, 10, "D3 5" );
	$worksheet->write( $count+27, 10, "D2 6" );
	$worksheet->write( $count+28, 10, "D1 7" );
	$worksheet->write( $count+29, 10, "D0 8" );
	$worksheet->write( $count+30, 10, "ACK 9" );
	
	$worksheet->write( $count+31, 10, "D7 1" );
	$worksheet->write( $count+32, 10, "D6 2" );
	$worksheet->write( $count+33, 10, "D5 3" );
	$worksheet->write( $count+34, 10, "D4 4" );
	$worksheet->write( $count+35, 10, "D3 5" );
	$worksheet->write( $count+36, 10, "D2 6" );
	$worksheet->write( $count+37, 10, "D1 7" );
	$worksheet->write( $count+38, 10, "D0 8" );
	$worksheet->write( $count+39, 10, "ACK 9" );
	
	$worksheet->write( $count+40, 10, "Stop" );
	$worksheet->write( $count+41, 10, "QMS" );
	$lineno++;
	$countnum++;
	$count+=45;
	$pmbus++;
}	
	
	
	