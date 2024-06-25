void MAGIC_excess()
{ 
  TString source = "BLLac_2020"; //source name
  TString infilename = "data/Output_flute.root"; //source path
  TFile* infile = new TFile(infilename);
  MHExcessEnergyTheta* theta = new MHExcessEnergyTheta();
  theta->Read("MHExcessEnergyTheta");
  TH1D* prhi = new TH1D();
  prhi = theta->GetHist()->ProjectionX();

  TString outfilename = "excess_" + source + ".root";
  TFile* outfile = new TFile(outfilename, "recreate");
  prhi->Write();
  outfile->Close();
  cout << "finished " << source << endl;
  // }
  return;
}
