void MAGIC_IRF()
{
  //To run this code you need to use root with MARS software.

  TString source = "BLLac_2020"; //source name
  TString infilename = "/home/rgrau/Desktop/EBL_analysis_Feb2018/BLLac2020/Output_flute_150_EPWL.root"; //path of the Output_flute.root file
  TFile* infile = new TFile(infilename);

  TH1D* hbckg = new TH1D();
  hbckg->Read("hEstBckgE");

  MHMcCollectionArea* collareaVsEest = new MHMcCollectionArea; 
  collareaVsEest->Read("collareaVsEest");

  MHEffectiveOnTime* effontime = new MHEffectiveOnTime;
  effontime->Read("MHEffectiveOnTime");

  TH2D* heot = (TH2D*) effontime->GetHEffOnPhiTheta().Clone("heot");

  collareaVsEest->CalcExposure(heot);

  Float_t minEtrue = collareaVsEest->GetHistExposureInEestBin(0)->GetXaxis()->GetXmin();
  Float_t maxEtrue = collareaVsEest->GetHistExposureInEestBin(0)->GetXaxis()->GetXmax();
  Int_t nbinsEtrue = collareaVsEest->GetHistExposureInEestBin(0)->GetNbinsX();

  Int_t nbinsEest = hbckg->GetNbinsX();
  Float_t minEest = hbckg->GetXaxis()->GetXmin();
  Float_t maxEest = hbckg->GetXaxis()->GetXmax();

  TAxis* EestAxis = (TAxis*) hbckg->GetXaxis();
  TAxis* EtrueAxis = (TAxis*) collareaVsEest->GetHistExposureInEestBin(0)->GetXaxis();

  TH2D* migmatrix = new TH2D("mig_matrix", "Exposure (m2 * s) vs. Etrue vs. Eest",
          nbinsEtrue, EtrueAxis->GetXbins()->GetArray(), 
          nbinsEest,  EestAxis->GetXbins()->GetArray());
  migmatrix->GetXaxis()->SetTitle("Etrue (GeV)");
  migmatrix->GetYaxis()->SetTitle("Eest (GeV)");

  for (Int_t j = 1; j <= nbinsEest; j++)
    {
      TH1D* slice = collareaVsEest->GetHistExposureInEestBin(j-1);
      for (Int_t i = 1; i <= nbinsEtrue; i++)
  {
    migmatrix->SetBinContent(i, j, slice->GetBinContent(i));
    migmatrix->SetBinError(i, j, slice->GetBinError(i));
  }
    }

  migmatrix->SetStats(0);
  migmatrix->Draw("zcol");
  gPad->SetLogx();
  gPad->SetLogy();
  gPad->SetLogz();
  gPad->SetGridx();
  gPad->SetGridy();

  TString outfilename = "fold_migmatrix_" + source + ".root";
  TFile* outfile = new TFile(outfilename, "recreate");
  migmatrix->Write();
  outfile->Close();
  cout << "finished " << source << endl;
  // }
  return;
}

