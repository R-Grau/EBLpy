#include <TROOT.h>
#include "Minuit2/Minuit2Minimizer.h"
//#include "Math/GSLMinimizer.h"
//#include "Math/GSLNLSMinimizer.h"
#include "Math/Functor.h"
#include <iostream>
#include <TApplication.h>
#include <TAxis.h>
#include <TCanvas.h>
#include <TCollection.h>
#include <TKey.h>
#include <TFile.h>
#include <TGraph.h>
#include <TGraph2D.h>
#include <TGraphErrors.h>
#include <TGraphAsymmErrors.h>
#include <TH1F.h>
#include <TH1D.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TLine.h>
#include <TMath.h>
#include <TMatrixDSym.h>
#include <TParameter.h>
#include <TVector.h>
#include <TStopwatch.h>
#include <TStyle.h>
#include <TString.h>
#include <TSystem.h>
#include <TVirtualFitter.h>

#include "MAnalysisProblems.h"
#include "MArgs.h"
#include "MGraphicsWizard.h"
#include "MHMcCollectionArea.h"
#include "MHEffectiveOnTime.h"
#include "MHExcessEnergyTheta.h"
#include "MLog.h"
#include "MLogManip.h"
#include "MMarsVersion.h"
#include "MReadEBLtau.h"
#include "MSpectralLikelihood.h"
#include "MStatusDisplay.h"

#define TEVTOERG 1.60218

using namespace std; 

//
// fold
//
// NOTE: you need to have ROOT compiled with the minuit2 optional package! (see "configure" executable in $ROOTSYS)
//
// This is a program to obtain a source's intrinsic spectral parameters through a forward folding procedure,
// maximizing the Poissonian likelihood of the  event distributions (ON and OFF) vs. Eest in the space of
// the spectral parameters. No assumption is made on the dependence of the background with energy, i.e. background
// (PoissonÇ) parameters in the different Eest bin are considered to be independent, and are treated as nuisance
// parameters.
// The program fold allows to set the source redshift and a scaling factor for the EBL density, as well as a spectral
// shape among power-law (PWL), Log-parabola (LP), power-law with exponential cut-off (EPWL), log parabola with
// exponential cut-off (ELP), and power-law with super-exponential cut-off (SEPWL). The LP, ELP and EPWL are forced
// to be concave (i.e. becoming softer as energy increases).
// The program reads as input the flute output, Output_flute.root (or the output from foam, the merger of flute outputs),
// from which both the MAGIC response function and the event distributions vs. Eest are read.
// 

//////////////////////////////////////////////////////////////////////////////////
//
// Some variables must be global because they are needed by Chi2() :
//
Double_t tauscale = 1.;  // just dummy initialization
Int_t npars = 0;          // Total number of parameters (just dummy initialization)
Int_t nparsIntrinsic = 0; // Number of parameters of intrinsic spectral function (just dummy initialization)
Int_t nparsEBL = 0;       // Number of EBL parameters (will be 0 or 1)


TStopwatch* chi2timer = 0;
MSpectralLikelihood* SpectralLikelihood;  // To calculate likelihood of a given spectrum, given a data set
MGraphicsWizard * gWizard;

Double_t performLikelihoodMaximization(TString inputfile,  Double_t redshift, TString model,
				       Double_t normalizationE,
				       Double_t scaleOpticalDepth = 1., TGraph2D* taugraph = 0,
				       MStatusDisplay* disp = 0, Bool_t WriteOutput = kFALSE,
				       Double_t* xpars = 0, Double_t* FitProb = 0,
				       Double_t fitLowE = -1., Double_t fitHighE = -1., Double_t BackgSystematics = 0.,
				       Double_t FermiE = -1., Double_t FermiIndex = 0., Double_t FermiDeltaIndex = 0.,
				       Double_t FermidFdE = -1., Double_t FermiDeltadFdE = -1., Double_t FermiCorrelation = 0.,
				       Double_t FermiFluxSystematics = 0.,
				       Bool_t SmoothMC=kFALSE,
				       Bool_t PropagateMCuncertainties=kTRUE);

Bool_t createFunction(TString funcname, TF1* fn, Double_t Enorm, Double_t*x, Double_t* step);
Bool_t createFunctionLog(TString funcname, TF1* fn, Double_t Enorm, Double_t*x, Double_t* step);

Float_t kLightScaleFactor = 1.0; // factor to simulate (if != 1.0) a miscalibration of MC and data. A value >1.0 means the MC total light
// throughput (including atmosphere) is higher (by that factor) than for the real observation.
Double_t kMinDisplayedSignificance;
Double_t kMinUsedSignificance;
Bool_t kIs_log_dFdE;
Float_t kPar1Max;  // Maximum value of fit parameter 1, i.e. the (negative) power-law index for PWL
Bool_t kMinos; // Calculate Minos errors

////////////////////////////////////////////////////////////
//
// The function below is the one to be minimized:
//
double Chi2(const double *x)
{
  //  chi2timer->Start(kFALSE);

  for (Int_t ipar = 0; ipar < nparsIntrinsic; ipar++)
    {
      if (TMath::IsNaN(x[ipar]))  // NOTE!!! Sometimes parameters become NaN during minimization!! It seems that just returning a high Chi2 value solves the issue...
       	{
       	  gLog << "WARNING!! Parameter " << ipar << " is NaN!" << endl;
	  //	  chi2timer->Stop();
       	  return 1.e6;
      	}
      SpectralLikelihood->SetIntrinsicParameter(ipar, x[ipar]);
    }

  if (nparsEBL > 0)
    tauscale = x[npars-1]; // This is for  the case nparsEBL == 1 only!

  SpectralLikelihood->SetEBLscale(tauscale);

  return SpectralLikelihood->CalcChi2();
}

////////////////////////////////////////////////////////////////////////////////
//
// Explain the usage of fold
//
static void Usage()
{
  gLog << all << endl;
  gLog << "Usage of fold:" << endl;
  gLog << "   fold [options]" << endl << endl;
  gLog << "Fold performs a maximum-likelihood Poissonian forward-folding fit to MAGIC spectra" << endl;
  gLog << "Options:" << endl;
  gLog << "   --inputfile=xxx.root : input file name (may be Output of flute or foam)" << endl;
  gLog << "   --function=PWL : function to be fitted (PWL, LP, EPWL, ELP, SEPWL, freeLP, CMBPWL)" << endl;
  gLog << "   --NormalizationE=xxx  : normalization energy for spectral functions (GeV)" << endl;
  gLog << "   --log=xxx.log : log file name" << endl;
  gLog << "   --redshift=xxx : source redshift. If < 0, a scan will be performed from 0 to the provided (absolute) value." << endl;
  gLog << "   --minEest=xxx  : (GeV) minimum estimated energy for fit." << endl;
  gLog << "   --maxEest=xxx  : (GeV) maximum estimated energy for fit. [Default range: all bins with events]" << endl;
  gLog << "   --minUsedSignificance=xxx  : minimum excess significance to use a bin in the fit (none by default)" << endl;
  gLog << "   --minDisplayedSignificance=xxx  : minimum excess significance to show a spectral point (2.0 by default)" << endl;
  gLog << "   --backgSystematics=xx : relative (gaussian) systematic uncertainty in background normalization" << endl;
  gLog << "   --EBLmodel=xxx : EBL template model: D11  (Dominguez 2011, default)" << endl <<
	  "                                        G12  (Gilmore 2012)" << endl <<
	  "                                        F08  (Franceschini 2008)" << endl <<
	  "                                        F18  (Franceschini 2018)" << endl <<
	  "                                        FI10 (Finke 2010)" << endl <<
	  "                                        H12  (Helgason 2012)" << endl <<
	  "                                        I13  (Inoue 2013)" << endl <<
	  "                                        K10  (Kneiske 2010)" << endl <<
	  "                                        S16  (Stecker 2016)" << endl;
  gLog << "   --EBLscale=xxx : scale factor for EBL optical depth w.r.t. the chosen template model" << endl;
  gLog << "   --FermiE=xxx : (GeV) additional photon index constraint from Fermi provided at this energy (pivot energy!)" << endl;
  gLog << "   --FermiIndex=xxx : photon index value from Fermi (at energy FermiE)" << endl;
  gLog << "   --FermiDeltaIndex=xxx : photon index uncertainty from Fermi (at energy FermiE)" << endl;
  gLog << "   --FermiSED=xxx : spectral energy density value from Fermi (at energy FermiE), in erg cm-2 s-1" << endl;
  gLog << "   --FermiDeltaSED=xxx : SED uncertainty from Fermi (at energy FermiE), in erg cm-2 s-1" << endl;
  gLog << "   --FermiCorrelation=xxx : correlation coefficient between Fermi flux (dF/dE) and photon index (index defined positive!)" << endl;
  gLog << "   --FermiFluxSystematics=xxx : relative (gaussian) systematic uncertainty in the Fermi flux value" << endl;
  gLog << "   --smoothMC : smooth the effective area from MC to avoid too strong influence of single MC events in case of very steep spectra" << endl;
  gLog << "   --ignoreMCuncertainties: ignore (set to 0) the uncertainties in the MC-derived response of the instrument" << endl;
  gLog << "   --LightScaleFactor=xxx: factor to simulate mismatch of MC and real data light scale. >1 means MC overestimates light throughput" << endl;
  gLog << "   --logdFdE : fitted function will be log(dF/dE) instead of dF/dE.  This improves convergence." << endl;
  gLog << endl;
  gLog << "   -h, --help: show this help" << endl;
  gLog << "   -b: Batch mode (no graphical output to screen)" << endl;
  gLog << "   -q: quit after finishing" << endl;
  gLog << "   --debug[=n] : set debug level to 1 [or n]" << endl;   
  gLog << endl;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// MAIN CODE STARTS HERE!!!
//
int main (int argc, char** argv)
{
  Double_t* xpars = 0;
  
  MArgs arg(argc, argv, kTRUE);
  if (arg.HasOnly("-h") || arg.HasOnly("--help"))
    {
      Usage();
      return 0;
    }


  if(!arg.HasOption("--log="))
    {
      TString defaultLogName = "Log_fold.log";
      gLog << inf << "Setting output log file to default (" << defaultLogName.Data() << ")." << endl;
      gLog.SetOutputFile(defaultLogName.Data(), 1);
    }
  else
    {
      gLog << inf << "Setting output log file to " << arg.GetString("--log=") << endl;
      gLog.Setup(arg);
    }
  arg.Print();

  TApplication app("fold", &argc, argv);

  // Search for "batch" and "quit" options:
  Bool_t kBatch = arg.HasOnlyAndRemove("-b");
  Bool_t kQuit = arg.HasOnlyAndRemove("-q");
  TString inputfile = arg.GetStringAndRemove("--inputfile=", "Output_flute.root");
  TString model = arg.GetStringAndRemove("--function=", "LP");
  Double_t normalizationE = arg.GetFloatAndRemove("--NormalizationE=", -1.);
  TString ebltemplate = arg.GetStringAndRemove("--EBLmodel=", "D11");
  Double_t scaleOpticalDepth = arg.GetFloatAndRemove("--EBLscale=", 1.);
  Double_t reds = arg.GetFloatAndRemove("--redshift=", 0.);
  Double_t eestmin = arg.GetFloatAndRemove("--minEest=", -1.); // if not supplied, Eest range will be determined automatically by MSpectralLikelihood
  Double_t eestmax = arg.GetFloatAndRemove("--maxEest=", -1.);
  kMinUsedSignificance = arg.GetFloatAndRemove("--minUsedSignificance=", -100.);
  kMinDisplayedSignificance = arg.GetFloatAndRemove("--minDisplayedSignificance=", 1.5);
  Double_t BackgSystematics = arg.GetFloatAndRemove("--backgSystematics=", 0.);

  // Are there additional constraints from Fermi? :
  Double_t FermiE = arg.GetFloatAndRemove("--FermiE=", -1.);
  Double_t FermiIndex = arg.GetFloatAndRemove("--FermiIndex=", -1.);
  Double_t FermiDeltaIndex = arg.GetFloatAndRemove("--FermiDeltaIndex=", -1.);
  Double_t FermiSED = arg.GetFloatAndRemove("--FermiSED=", -1.);
  Double_t FermiDeltaSED = arg.GetFloatAndRemove("--FermiDeltaSED=", -1.);
  Double_t FermiCorrelation = arg.GetFloatAndRemove("--FermiCorrelation=", 0.);
  Double_t FermiFluxSystematics = arg.GetFloatAndRemove("--FermiFluxSystematics=", 0.);
  
  // Convert SED values to differential energy spectrum:
  Double_t FermidFdE = FermiSED / TEVTOERG / pow(1.e-3*FermiE, 2.); // cm^-2 s^-1 TeV^-1
  Double_t FermiDeltadFdE = FermiDeltaSED / TEVTOERG / pow(1.e-3*FermiE, 2.); // cm^-2 s^-1 TeV^-1

  kLightScaleFactor = arg.GetFloatAndRemove("--LightScaleFactor=", 1.);
  kIs_log_dFdE = arg.HasOnlyAndRemove("--logdFdE");   // (use log(dFdE) in fit, instead of dFdE )

  kPar1Max = arg.GetFloatAndRemove("--Par1Max=", 0.);
  kMinos = ! arg.HasOnlyAndRemove("--NoMinos");
  if (reds < 0.)
    kMinos = kFALSE; // To speed up the redshift scan

  gDebug = arg.HasOption("--debug=") ? arg.GetIntAndRemove("--debug=") : 0;
  if (gDebug == 0 && arg.HasOnlyAndRemove("--debug"))
    gDebug=1;

  Bool_t SmoothMC = arg.HasOnlyAndRemove("--smoothMC");
  if (SmoothMC)
    gLog << endl << "NOTE: smoothing of the MC-derived IRF is set to ON by the user." << endl;
    
  Bool_t PropagateMCuncertainties = ! arg.HasOnlyAndRemove("--ignoreMCuncertainties");
  if (!PropagateMCuncertainties)
    gLog << endl << "NOTE: the user has chosen to ignore the uncertainties in the MC-derived IRF." << endl;


  if (arg.GetNumOptions()>0)
    {
      gLog << warn << "WARNING - Unknown commandline options..." << endl;
      arg.Print("options");
      gLog << endl;
      return MAnalysisProblems::kWrongArguments;
    }


  gWizard = new MGraphicsWizard(0.05, 42);

  gROOT->SetBatch(kBatch);
  

  // Open the status display:
  MStatusDisplay* disp = new MStatusDisplay(960, 720);
  disp->SetWindowName("fold - forward-folding Poissonian likelihood spectral fitting");

  // Read in the EBL data:
  TGraph2D* taugraph = new TGraph2D;
  MReadEBLtau eblreader;
  Bool_t eblok = eblreader.ReadModel(ebltemplate, taugraph);
  if (!eblok)
    return MAnalysisProblems::kWrongArguments;

  Double_t final_chi2 = 0.;

  // Below are the calls which actually carry out the likelihood maximization:

  if (reds < 0.)  // Redshift scan option
    {
      TGraph* chi2_vs_z = new TGraph;
      TGraph* conflev_vs_z = new TGraph;
      TGraph* par1_vs_z = new TGraph;

      chi2_vs_z->SetName("chi2_vs_z");
      conflev_vs_z->SetName("conflev_vs_z");
      par1_vs_z->SetName("par1_vs_z");

      gROOT->SetBatch(kTRUE);

      if (model == "PWL")
	{
	  xpars = new Double_t[2];
	  xpars[0] = -1.;  // => performLikelihoodMaximization will look for reasonable starting params
	}

      for (Float_t zz = 0.; zz < -reds+1.e-6; zz += -reds/10.)  // Scan in redshift!
	{

	  Float_t chi2 = performLikelihoodMaximization(inputfile, zz, model, normalizationE, scaleOpticalDepth, taugraph, 0, kFALSE, xpars, 0,
						       eestmin, eestmax, BackgSystematics, FermiE, FermiIndex, FermiDeltaIndex, FermidFdE,
						       FermiDeltadFdE, FermiCorrelation, FermiFluxSystematics, SmoothMC,
						       PropagateMCuncertainties);

	  gLog << "Redshift: " << Form("%.2f", zz) << ",  Chi2: " << Form("%.3f", chi2) << endl;
	  chi2_vs_z->SetPoint(chi2_vs_z->GetN(), zz, chi2);
	  if (xpars)
	    par1_vs_z->SetPoint(par1_vs_z->GetN(), zz, xpars[1]);
	}

      Float_t minchi2 = 1.e6;
      for (Float_t zz = 0.; zz < -reds+1.e-6; zz += -reds/100.)
	{
	  Float_t dummy = chi2_vs_z->Eval(zz, 0, "S");
	  if (dummy > minchi2)
	    continue;
	  minchi2 = dummy;
	}

      // The plotted chi2 is the profile likelihood -2logL relative to the absolute (unconstrained) maximum likelihood, i.e. all bin contents exactly "predicted" by the model
      // Following Rolke, López & Conrad, arXiv:0403059 we use it to obtain an upper limit to the redshift. The delta_chi2 relative to the chi2 minimum (within z>=0) is
      // converted into confidence level of the upper limit. We do this in smaller steps (from interpolated Chi2 graph) otherwise it is too discretized.

      for (Float_t zz = 0.; zz < -reds+1.e-6; zz += -reds/100.)
	// fabs needed because, due to rounding, chi2-minchi2 may be infinitesimally negative!
	conflev_vs_z->SetPoint(conflev_vs_z->GetN(), zz,
			       1.-0.5*TMath::Prob(fabs(chi2_vs_z->Eval(zz, 0, "S")-minchi2), 1)); // Factor 0.5* because it is an UL (1-sided interval)

      gROOT->SetBatch(kBatch);

      TCanvas& canvchi2 = disp->AddTab("Chi2 vs. z");
      gWizard->WhiteBack(canvchi2);
      canvchi2.SetGridx();
      canvchi2.SetGridy();
      chi2_vs_z->Draw("ac");
      chi2_vs_z->GetXaxis()->SetTitle("redshift");
      chi2_vs_z->GetYaxis()->SetTitle("Chi2");

      TCanvas& canvconflev = disp->AddTab("1-sided CL vs. z");
      gWizard->WhiteBack(canvconflev);
      canvconflev.SetGridx();
      canvconflev.SetGridy();

      //      conflev_vs_z->SetLineStyle(2);
      //      conflev_vs_z->Draw("al");
      //      conflev_vs_z->GetXaxis()->SetTitle("redshift U.L.");
      //      conflev_vs_z->GetYaxis()->SetTitle("1-sided C.L.");
      Int_t ii;
      for (ii = 0; ii < conflev_vs_z->GetN(); ii++) // Find the redshift point at which the minimum Chi2 is reached:
	if (conflev_vs_z->GetY()[ii] == TMath::MinElement(conflev_vs_z->GetN(), conflev_vs_z->GetY()))
	  break;
      TGraph* gUL = new TGraph(conflev_vs_z->GetN()-ii, conflev_vs_z->GetX()+ii, conflev_vs_z->GetY()+ii);
      gUL->SetName("gUL");
      gUL->SetTitle("C.L. vs. redshift upper limit");
      gUL->GetXaxis()->SetTitle("redshift U.L.");
      gUL->GetYaxis()->SetTitle("1-sided C.L.");

      gUL->SetLineWidth(2);
      gUL->Draw("al");

      TLine lin;
      lin.DrawLine(0., 0.95, -reds, 0.95);

      if (xpars)
	{
	  TCanvas& canvpar1 = disp->AddTab("Par1");
	  gWizard->WhiteBack(canvpar1);
	  canvpar1.SetGridx();
	  canvpar1.SetGridy();
	  par1_vs_z->GetXaxis()->SetTitle("redshift");
	  par1_vs_z->GetYaxis()->SetTitle("Par1 (index of PWL)");
	  par1_vs_z->Draw("al");
	}

      disp->SaveAsRoot("Status_fold_zscan.root");

      TFile fout("Output_fold_zscan.root", "recreate");
      conflev_vs_z->Write();
      conflev_vs_z->Write();
      gUL->Write();
      chi2_vs_z->Write("chi2_vs_z");
      fout.Close();
    }

  else   // Normal, fixed redshift fit:
    final_chi2 = performLikelihoodMaximization(inputfile, reds, model, normalizationE, scaleOpticalDepth, taugraph, disp, kTRUE, xpars, 0,
					       eestmin, eestmax, BackgSystematics, FermiE, FermiIndex, FermiDeltaIndex, FermidFdE,
					       FermiDeltadFdE, FermiCorrelation, FermiFluxSystematics, SmoothMC, PropagateMCuncertainties);

  if (kQuit || kBatch || final_chi2 < 0.)
    delete disp;
  else
    {
      disp->SetBit(MStatusDisplay::kExitLoopOnExit);
      disp->SetBit(MStatusDisplay::kExitLoopOnClose);
      // Wait until the user decides to exit the application:
      app.Run(kFALSE);
    }

  if (final_chi2 < 0.)
    return -1;
  
  return 0;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// The actual likelihood maximization is performed below. This can also be called from e.g. a ROOT macro by doing
// gROOT->LoadMacro("$MARSSYS/fold.cc+")
//
// 
// The function performLikelihoodFit calculates the chi2 (NOT normalized) of the (forward-folding) best-fit of the intrinsic
// spectral shape "model" (one of the arguments) to the differential energy spectrum found in the Output_flute.root file.
//
// If xpars != NULL, it can contain the starting intrinsic spectral parameters for the minimization (will be ignored if the first
// parameter, usually the flux normalization, is <= 0).
// If the pointer xpars is not NULL, the result of the minimization will be written in the array xpars (including EBL density in
// the last filled entry, in case it was fitted).
//
// If scaleOpticalDepth < 0, then the EBL density scaling will be treated as a free parameter and fitted.
//

Double_t performLikelihoodMaximization(TString inputfile, Double_t redshift, TString model, Double_t normalizationE, Double_t scaleOpticalDepth,
				       TGraph2D* taugraph, MStatusDisplay* disp, Bool_t WriteOutput, Double_t* xpars,
				       Double_t* FitProb, Double_t fitLowE, Double_t fitHighE, Double_t BackgSystematics,
				       Double_t FermiE, Double_t FermiIndex, Double_t FermiDeltaIndex, 
				       Double_t FermidFdE, Double_t FermiDeltadFdE, Double_t FermiCorrelation, Double_t FermiFluxSystematics,
				       Bool_t SmoothMC, Bool_t PropagateMCuncertainties)
{
  Bool_t deletedisp = kFALSE;
  if (!disp)  // if not provided, create it here (but remember to delete it)
    {
      disp = new MStatusDisplay(960, 720);
      disp->SetWindowName("fold - forward-folding Poissonian likelihood spectral fitting");
      deletedisp = kTRUE;
    }

  //  chi2timer = new TStopwatch;

  SpectralLikelihood = new MSpectralLikelihood;

  SpectralLikelihood->SetSmoothMC(SmoothMC);
  SpectralLikelihood->SetPropagateMCuncertainties(PropagateMCuncertainties);
  SpectralLikelihood->SetLightScaleFactor(kLightScaleFactor);

  SpectralLikelihood->SetMinPointSignificance(kMinDisplayedSignificance);
  SpectralLikelihood->SetMinSignificance(kMinUsedSignificance);

  if (fitLowE > 0. || fitHighE > 0.)
    SpectralLikelihood->SetUsedEestRange(fitLowE, fitHighE);

  if (BackgSystematics > 0.)
    SpectralLikelihood->SetBackgroundSystematics(BackgSystematics);

  if (!SpectralLikelihood->ReadInputFile(inputfile))
    exit(-1);


  SpectralLikelihood->SetRedshift(redshift);

  if (FermiE >0. && FermiIndex > 0. && FermiDeltaIndex > 0. && FermidFdE > 0. && FermiDeltadFdE > 0.)
    SpectralLikelihood->SetFermiConstraint(FermiE, FermidFdE, FermiDeltadFdE, FermiIndex, FermiDeltaIndex,
					   FermiCorrelation, FermiFluxSystematics);

  
  
  TGraphAsymmErrors* DiffEspectrum = SpectralLikelihood->GetDiffEspectrum(); // This is the Flute spectrum - will be used to get rough starting parameters


  Double_t normE;
  if (normalizationE < 0.)
    {
      normE = SpectralLikelihood->GetAverageEnergy();
      gLog << inf << "Energy normalization set at " << Form("%.1f", normE) << " GeV" << endl;
    }
  else
    {
      normE = normalizationE;
      gLog << inf << "Energy normalization FIXED BY USER at " << Form("%.1f", normE) << " GeV" << endl;
    }

  // Define function here (incl vectors x with initial pars and step):
  // For now: max 4 parameters for the intrinsic spectrum + (optionally) 1 EBL parameter (=the EBL density)
  //
  Double_t x[5]    = {0., 0., 0., 0., 0.};
  Double_t step[5] = {0., 0., 0., 0., 0.};

  // Initial parameters (in x[], set by createFunction) will be used only if not supplied by user through the argument xpars

  TF1* func = new TF1;
  Bool_t isok = kIs_log_dFdE? createFunctionLog(model, func, normE, x, step) : createFunction(model, func, normE, x, step);
  if (! isok)
    {
      gLog << err << "Unknown spectral model! : " << model << endl;
      if (deletedisp)
	delete disp;

      delete SpectralLikelihood;
      
      return -1.;
    }
  else
    gLog << inf << "Function expression: " << (kIs_log_dFdE? "log(dF/dE)" : "dF/dE") << " = " << func->GetExpFormula() << endl;


  // Note: the Name of the function func is model.Data(), that is "LP", or "PWL" or whatever. 

  SpectralLikelihood->SetIntrinsicSpectrum(func, kIs_log_dFdE);
  
  nparsIntrinsic = func->GetNpar();  // Number of parameters of intrinsic spectral function
  npars = nparsIntrinsic;  // npars stores the total number of free parameters (we can include free EBL parameters below)
  
  if (scaleOpticalDepth < 0.)
    {
      nparsEBL = 1;
      npars += 1.;  // Additional free parameter is the EBL density (=scaling factor of optical depth w.r.t. EBL model)
      // NOTE!! From now on npars is *not* the number of parameters of the intrinsic spectral function "func"! It includes also the EBL parameter!
      x[npars-1] = 0.; // starting value for EBL density
      tauscale = x[npars-1]; // just the starting value.
      step[npars-1] = 0.1; // Units are "nominal EBL density in the Dominguez model" (or whatever model is used)
    }
  else
    tauscale = scaleOpticalDepth; // fixed EBL  (with given optical depth scaling w.r.t. used EBL model).


  SpectralLikelihood->SetTau_vs_E(taugraph);
  TGraph* tauvsE = SpectralLikelihood->GetTau_vs_E();
  
  // Draw optical depth vs. E and redshift:
  if (redshift > 0.)
    {
      TCanvas& canvtau = disp->AddTab("EBL");
      gWizard->WhiteBack(canvtau);
      canvtau.SetLogz();
      gROOT->cd(); // needed for the clone
      TGraph2D* dummy = (TGraph2D*) taugraph->DrawClone("cont4,z");
      dummy->GetHistogram()->GetYaxis()->SetTitle("redshift");
      dummy->GetHistogram()->GetXaxis()->SetTitle("log10(E/TeV)");
      dummy->GetHistogram()->SetMinimum(0.01);

      TCanvas& canvtau2 = disp->AddTab("EBL2");
      gWizard->WhiteBack(canvtau2);
      canvtau2.SetLogy();
      canvtau2.SetGridx();
      canvtau2.SetGridy();
      TGraph* dummy2 = (TGraph*) tauvsE->DrawClone("al");
      dummy2->SetName(Form("z=%.3f", redshift));
      dummy2->GetHistogram()->GetYaxis()->SetTitle("tau");
      dummy2->GetHistogram()->GetXaxis()->SetTitle("log10(E/TeV)");
    }
  else // redshift 0 => null optical depth for any E
    {
      for (Int_t i = 0; i < tauvsE->GetN(); i++)
	tauvsE->SetPoint(i, tauvsE->GetX()[i], 0.);
    }


  // Has the user supplied starting parameters?? If so, use them:
  Bool_t inputok = kFALSE;
  if (xpars != 0)
    {  
      if (xpars[0] > 0.)  // Otherwise these cannot be valid starting params!
	{
	  inputok = kTRUE;
	  for (Int_t ipar = 0; ipar < npars; ipar++)
	    x[ipar] = xpars[ipar];
	}
    }

  TVirtualFitter *fitter = 0; // to be used later for determining decorrelation energy.

  if (!inputok)  // => means user did not specify valid starting parameters. Hence we look for them.
    {
      // Here we will make a normal ROOT fit to deabsorbed Flute spectral points:
      //
      // De-absorb the (scaled) ebl:
      TGraphAsymmErrors* deabsorbed_dfde = (TGraphAsymmErrors*)DiffEspectrum->Clone("deabsorbed_dfde");
      Double_t *xdfde = deabsorbed_dfde->GetX(); 
      Double_t *ydfde = deabsorbed_dfde->GetY(); 
      Double_t *eydfde_low = deabsorbed_dfde->GetEYlow(); 
      Double_t *eydfde_high = deabsorbed_dfde->GetEYhigh(); 
      Int_t idfde;
      for (idfde = 0; idfde < deabsorbed_dfde->GetN(); idfde++)
	{
	  if (kIs_log_dFdE)
	    {
	      Double_t newy = tauscale*tauvsE->Eval(log10(xdfde[idfde])-3.) + log(ydfde[idfde]);
	      deabsorbed_dfde->SetPointError(idfde, 0., 0.,
					     eydfde_low[idfde]/ydfde[idfde],
					     eydfde_high[idfde]/ydfde[idfde]);
	      deabsorbed_dfde->SetPoint(idfde, xdfde[idfde], newy);
	    }
	  else
	    {
	      Double_t newy = TMath::Exp(tauscale*tauvsE->Eval(log10(xdfde[idfde])-3.)) * ydfde[idfde];
	      deabsorbed_dfde->SetPointError(idfde, 0., 0.,
					     newy*eydfde_low[idfde]/ydfde[idfde],
					     newy*eydfde_high[idfde]/ydfde[idfde]);
	      deabsorbed_dfde->SetPoint(idfde, xdfde[idfde], newy);
	    }
	}

      if (FermiE > 0. && FermiIndex > 0. && FermiDeltaIndex > 0. && FermidFdE > 0. && FermiDeltadFdE > 0.) // Add one Fermi point:
	{
	  if (kIs_log_dFdE)
	    {
	      deabsorbed_dfde->SetPoint(idfde, FermiE, log(FermidFdE));
	      deabsorbed_dfde->SetPointError(idfde, 0., 0., FermiDeltadFdE/FermidFdE, FermiDeltadFdE/FermidFdE);
	    }
	  else
	    {
	      deabsorbed_dfde->SetPoint(idfde, FermiE, FermidFdE);
	      deabsorbed_dfde->SetPointError(idfde, 0., 0., FermiDeltadFdE, FermiDeltadFdE);
	    }
	}

      func->SetParameter(0, deabsorbed_dfde->Eval(normE)); // Better starting value!

      if (func->GetNpar() > deabsorbed_dfde->GetN()) // More parameters than points!! Fit only first parameters:
	{
	  for (Int_t ipar = deabsorbed_dfde->GetN(); ipar < func->GetNpar(); ipar++)
	    func->FixParameter(ipar, func->GetParameter(ipar));
	}

      TFitResultPtr resfit;
      // Fit intrinsic spectrum:
      resfit = deabsorbed_dfde->Fit(func,"R0,B");
      if (func->GetProb() < 1.e-5) // Try robust fit
	resfit = deabsorbed_dfde->Fit(func,"R0,B,ROB");

      for (Int_t ipar = 0; ipar < func->GetNpar(); ipar++)
	func->ReleaseParameter(ipar);
      
      fitter = TVirtualFitter::GetFitter();
      
      Int_t fitStat = resfit;
      gLog << inf << "Initial fit status: " << fitStat << "   ; Prob: " << func->GetProb() << endl;
  
      for (Int_t ipar = 0; ipar < nparsIntrinsic; ipar++)
	x[ipar] = func->GetParameter(ipar);
    }

  gLog << inf << "Chi2 before Likelihood maximization: " << Chi2(x) << endl;

  
  TH2D* EffTimevsAzZd = SpectralLikelihood->GetEffTimevsAzZd();
  Double_t efftime = EffTimevsAzZd->Integral(); // Total obs. time

  MHMcCollectionArea* aeffEestZd = SpectralLikelihood->GetAeffEestZd();  // => as read from the input file, i.e. assuming the tentative spectrum used in flute
  TH1D* aeffEestOriginal = aeffEestZd->GetHistCoarse()->ProjectionX("aeffEestOriginal", 1, 1, "e");  // in m2
  aeffEestOriginal->SetDirectory(0);

  TH1D* hobserved_events = SpectralLikelihood->GetHobserved_events();
  TH1D* hEstBckgE = SpectralLikelihood->GetHnormalized_off_events();

  
  TCanvas& canv1 = disp->AddTab("OFF");
  gWizard->WhiteBack(canv1);
  canv1.SetLogx();
  canv1.SetLogy();
  canv1.SetGridx();
  canv1.SetGridy();
  canv1.SetLeftMargin(0.15);
  canv1.SetBottomMargin(0.15);
  hEstBckgE->DrawCopy("e");
  
  Int_t firstEbin = SpectralLikelihood->GetFirstBin();
  Int_t lastEbin  = SpectralLikelihood->GetLastBin();
 

  // Draw range in the background histogram:
  TLine* lin = new TLine;
  lin->SetLineWidth(2);
  lin->SetLineStyle(2);
  lin->SetLineColor(4);

  lin->DrawLine(hEstBckgE->GetXaxis()->GetBinLowEdge(firstEbin), hEstBckgE->GetMinimum(), hEstBckgE->GetXaxis()->GetBinLowEdge(firstEbin), hEstBckgE->GetMaximum());
  lin->DrawLine(hEstBckgE->GetXaxis()->GetBinUpEdge(lastEbin),   hEstBckgE->GetMinimum(), hEstBckgE->GetXaxis()->GetBinUpEdge(lastEbin),   hEstBckgE->GetMaximum());  


  gLog << inf << endl << "Used energy range for the fit: " << hEstBckgE->GetXaxis()->GetBinLowEdge(firstEbin) << " to " 
       << hEstBckgE->GetXaxis()->GetBinUpEdge(lastEbin) << " GeV" << endl << endl; 


  //  ROOT::Math::MinimizerOptions::SetDefaultStrategy(2);  // DOES NOT SEEM TO HELP AT ALL!!!!

  // IN ORDER TO BE ABLE TO CALCULATE CONFIDENCE BANDS FOR THE LIKELIHOOD FIT, IT MAY BE BETTER TO USE
  // TVirtualFitter or TFitterMinuit...
  //
  //    TFitterMinuit min;
  // or
  //    TVirtualFitter::SetDefaultFitter("Minuit2");
  //    TVirtualFitter *min = TVirtualFitter::Fitter(0, 500); // max 500 parameters
  //
  // TO BE INVESTIGATED!!
  //
  

  ROOT::Minuit2::Minuit2Minimizer min ( ROOT::Minuit2::kMigrad );
  min.SetMaxFunctionCalls(10000);
  min.SetMaxIterations(1000);
  min.SetTolerance(0.01);
//  min.SetTolerance(0.1);
//  min.SetTolerance(1.);

  min.SetPrintLevel(5/*1*/);
  ROOT::Math::Functor f (&Chi2, npars); 
  min.SetFunction(f);
  TString varname[6]={"a", "b", "c", "d", "e", "f"};
  if (nparsEBL > 0)
    varname[npars-1] = "EBLdensity";
  
  for (Int_t ipar = 0; ipar < npars; ipar++)
    min.SetVariable(ipar, varname[ipar].Data(), x[ipar], step[ipar]);

  if (nparsEBL > 0)
    for (Int_t ieblpar = npars-nparsEBL; ieblpar < npars; ieblpar++)
      min.SetVariableLowerLimit(ieblpar, 0.);

  // Avoid problems with very steep (un-physical) functions which may result in low chi2  via a curious procedure: give huge
  // weight to low true energies, so that single bins in the MC exposure histograms with just one event (100% error in exposure)
  // dominate the prediction in the number of gammas in all bins of estimated energy: a huge number, but with 100% error, so
  // essentially compatible with any real data!
  min.SetVariableLimits(1, -6., kPar1Max);

  if (model == "freeLP")
    min.SetVariableLimits(2,  -10., 10.);
  else if (model == "pileupEPWL")
    min.SetVariableLowerLimit(2, 10.);  // NOTE! : to set 2-sided limits use SetVariableLimits, NOT Set...Upper   and then Lower => only the 2nd will work!
  else if (model == "freeEPWL")
    min.SetVariableLowerLimit(2, 0.1);

  // EXECUTE THE MINIMIZATION:
  
  Bool_t converged = min.Minimize();

  if (!converged)
    {
      gLog << endl << err << "ERROR, minimization did not converge! Status = " << min.Status() << endl << endl;
      if (deletedisp)
	{
	  delete disp;
	  delete lin;
	}

      delete SpectralLikelihood;
      
      return -1.;
    }

  
  //Set parameter uncertainties to 0 to avoid confusion.
  //func->GetParError(ipar) will return the uncertainties obtained by the preliminary fit,
  //and not those from the Minuit2 minimization.
  for (Int_t ipar = 0; ipar <= nparsIntrinsic-1; ipar++)
  func->SetParError(ipar, 0);
  

  // Calculate MINOS errors for spectral function parameters:
  Double_t* deltaparlow  = new Double_t[npars-nparsEBL];
  Double_t* deltaparhigh = new Double_t[npars-nparsEBL];
  if (kMinos)
    for (Int_t ipar = 0; ipar < nparsIntrinsic; ipar++)
      min.GetMinosError(ipar, deltaparlow[ipar], deltaparhigh[ipar]);

  gLog << inf << "Chi2 = " << Chi2(min.X()) << endl;

  Int_t ndof = lastEbin - firstEbin + 1 - npars;
  if (FermiE >0. && FermiIndex > 0. && FermiDeltaIndex > 0.)
    ndof += 1; // (one additional "point", i.e. Fermi's slope at a given energy)

    
  // TCanvas* canv_area = new TCanvas("canv_area", "", 400, 600);
  // canv_area->SetLogx();
  // canv_area->SetLogy();
  // canv_area->SetGridx();
  // canv_area->SetGridy();
  // aeffEestOriginal->SetStats(0);
  // aeffEestOriginal->Draw("e");
  
  // Double_t *covmatrix = new Double_t[npars*npars];
  // min.GetCovMatrix(covmatrix);
  // for (Int_t i = 0; i < npars; i++)
  //   for (Int_t j = 0; j < npars; j++)
  //     gLog << inf << "cov(" << i << "," << j << "): " << covmatrix[i+npars*j] << endl;

  TMatrixDSym corrmatrix(npars);
  for (Int_t i = 0; i < npars; i++)
    for (Int_t j = 0; j < npars; j++)
      {
	corrmatrix(i,j) = min.Correlation(i, j);
	gLog << inf << "Correlation (" << i << "," << j << ") : " << min.Correlation(i, j) << endl;
      }
  
  gLog << inf << endl;
  for (Int_t i = 0; i < npars; i++)
    gLog << inf << "Global correlation of parameter " << i << " = " << min.GlobalCC(i) << endl;
  gLog << endl;

  const double *xs = min.X();

  if (xpars != 0)
    for (Int_t i = 0; i < npars; i++)
      xpars[i] = xs[i];

  gLog << inf << "Function expression: " << (kIs_log_dFdE? "log(dF/dE)" : "dF/dE") << " = " << func->GetExpFormula() << endl;

  gLog << inf << "Intrinsic spectral parameters:    ";
  for (Int_t i = 0; i < nparsIntrinsic-1; i++)
    gLog << inf << xs[i] << ", ";
  gLog << inf << xs[nparsIntrinsic-1] << endl;

  gLog << inf << "Intr. spec. param. uncertainties (default ones from Migrad): ";
  const double *delta_xs = min.Errors();
  for (Int_t i = 0; i < nparsIntrinsic-1; i++)
    gLog << inf << delta_xs[i] << ", ";
  gLog << inf << delta_xs[nparsIntrinsic-1] << endl;

  gLog << endl;
  if (kMinos)
    {
      gLog << inf << "Intrinsic spec. parameters (with MINOS uncertainties): " << endl;
      for (Int_t i = 0; i < nparsIntrinsic; i++)
	gLog << inf << "parameter #" << i << "= " << xs[i] << "   +" << deltaparhigh[i] << "   " << deltaparlow[i] << endl;
      gLog << endl;
    }

  //
  // NOTE NOTE NOTE NOTE!! :
  //
  // Now we want to obtain the confidence interval (vs. E) of the fitted spectrum, in order to find the decorrelation energy.
  // Unfortunately it seems there is no automatic way to do it in ROOT for the poissonian likelihood maximization we have used
  // in our fit. The method below will actually do it for the preliminary chi2 fit that we made to find starting parameters.
  // Note that even if we initialized fitter here with TVirtualFitter::GetFitter the result will be the same: the fitter will
  // still be that of the previous call to a *->Fit  ROOT function, and _not_ something related to the likelihood maximization
  // we just did.
  // For this reason, we will NOT plot this confidence intervals, they do not correspond to the poissonian likelihood fit that
  // we have performed.
  //

  TGraphErrors* confint = 0;
  TGraphErrors* confintSED = 0;
  Double_t decorrelE = -1.;
	
  if (fitter)
    {
      confint = new TGraphErrors;
      confint->SetName("ConfidenceInterval");
      Int_t ipoint = 0;
      for (Double_t xen = log10(30.); xen < log10(30000.); xen += 0.01)
	confint->SetPoint(ipoint++, pow(10.,xen), 0);

      fitter->GetConfidenceIntervals(confint, 0.68);

      Double_t* ciey = confint->GetEY();
      Double_t* cix  = confint->GetX();
      Double_t* ciy  = confint->GetY();
      Double_t minrelwidth = 1.e6;
      
      for (ipoint = 0; ipoint < confint->GetN(); ipoint++)
	{
	  Double_t eval;
	  if (kIs_log_dFdE)
	    {
	      eval = ciey[ipoint];
	    }
	  else
	    {
	      // Check width of confidence interval in logarithmic scale (needs to skip points with negative error bar below 0)
	      if (ciey[ipoint] > ciy[ipoint])
		continue;
	      eval = (ciy[ipoint]+ciey[ipoint])/(ciy[ipoint]-ciey[ipoint]);
	    }

	  if (eval > minrelwidth) 
	    continue;
	  minrelwidth = eval;
	  decorrelE = cix[ipoint];
	}
      // gLog << inf << "Decorrelation energy: " << Form("%.2f", decorrelE) << " GeV. Used normalization energy: " 
      // 	   << Form("%.2f", normE) << " GeV." << endl;
      // if (fabs(decorrelE-normE) > 0.1)
      // 	gLog << "          Use --NormalizationE=" << Form("%.2f", decorrelE) << " if you want flux normalization at the decorrelation energy." << endl;
      
      confintSED = new TGraphErrors;
      confintSED->SetName("SED_ConfidenceInterval");
      for (Int_t i = 0; i < confint->GetN(); i++)
	{
	  Double_t xen = confint->GetX()[i];
	  if (kIs_log_dFdE)
	    {
	      confintSED->SetPoint(i, xen, xen*xen*1.e-6*exp(confint->GetY()[i]));
	      confintSED->SetPointError(i, 0.,
					xen*xen*1.e-6*(exp(confint->GetY()[i]+confint->GetEY()[i])-exp(confint->GetY()[i])) );
	    }
	  else
	    {
	      confintSED->SetPoint(i, xen, xen*xen*1.e-6*confint->GetY()[i]);
	      confintSED->SetPointError(i, 0., xen*xen*1.e-6*confint->GetEY()[i]);
	    }
	}

      //
      // NOTE: Besides not corresponding to fold's maximum likelihood fit, as far as I know this confidence interval comes from
      // extrapolation of the function using the fit parameters and uncertainties in linear approximation from the best-fit params. Hence
      // it will only work fine for x (=energy) values far from the "bulk" of the data points if the function is really linear in its 
      // parameters. When kIs_log_dFdE==kFALSE, none of our functions, the way they are written, is linear in all its parameters.
      // For example, one expects the confidence interval for a power law fit to be, in log log representation (i.e.
      // if kIs_log_dFdE==kTRUE), a kind of "butterfly", with the upper and lower limits of the confidence interval extending
      // straight to high and low energies. But the one calculated here when kIs_log_dFdE==kFALSE (i.e the fitted function is dF/dE)
      // would instead be curved, because it would extrapolate the linear approximation of the power law (in lin - lin
      // representation). In any case, this should be no issue in what refers to the decorrelation energy, which is within
      // the range of the measured spectral points.
      //      
    }
  
  Double_t errlow = 0.;
  Double_t errup = 0.;
  if (nparsEBL > 0)
    {
      min.GetMinosError(npars-1, errlow, errup);
      gLog << inf << "EBL density = " << Form("%.3f",xs[npars-1]) << " (" << Form("%.3f", errlow) << ") (+" << Form("%.3f", errup) << ")" << endl;
    }
    
  Double_t chisquare = Chi2(xs);
  Double_t fprobability = TMath::Prob(chisquare, ndof);
  gLog << inf << "Final Chi2: " << chisquare << " / " << ndof << endl;
  gLog << inf << "Prob = " << fprobability << endl;

  if (FitProb != 0)
    *FitProb = fprobability;
  
  gLog << inf << "Bins used in fit: from " << firstEbin << " to " << lastEbin << endl;


  if (deletedisp && gROOT->IsBatch() && !WriteOutput)
    {
      delete disp;
      delete lin;

      delete SpectralLikelihood;
      if (confint)
	delete confint;
      if (confintSED)
	delete confintSED;
      return chisquare;
    }


  //  Execute Chi2 one more time, but with histogram-filling option on:
  SpectralLikelihood->CalcChi2(kTRUE);

  TF1* abs_spec = SpectralLikelihood->GetAbsorbedSpectrum();


  canv1.cd();
  TH1D* hexpected_background = SpectralLikelihood->GetHexpected_background();
  hexpected_background->SetLineColor(2);
  hexpected_background->DrawCopy("same,histo");

  TCanvas& canv2 = disp->AddTab("ON");
  gWizard->WhiteBack(canv2);
  canv2.SetLogx();
  canv2.SetLogy();
  canv2.SetGridx();
  canv2.SetGridy();
  canv2.SetLeftMargin(0.15);
  canv2.SetBottomMargin(0.15);

  TH1D* hexpected_events = SpectralLikelihood->GetHexpected_events();
  hexpected_events->GetYaxis()->SetLabelSize(0.05);
  hexpected_events->GetYaxis()->SetTitleSize(0.05);
  hexpected_events->SetStats(0);
  hexpected_events->SetLineColor(2);
  hexpected_events->DrawCopy("e1");

  hobserved_events->SetLineColor(1);
  hobserved_events->DrawCopy("same,e");

  lin->DrawLine(hexpected_events->GetXaxis()->GetBinLowEdge(firstEbin), hexpected_events->GetMinimum(), hexpected_events->GetXaxis()->GetBinLowEdge(firstEbin), hexpected_events->GetMaximum());
  lin->DrawLine(hexpected_events->GetXaxis()->GetBinUpEdge(lastEbin),   hexpected_events->GetMinimum(), hexpected_events->GetXaxis()->GetBinUpEdge(lastEbin),   hexpected_events->GetMaximum());

  TLatex* lat = new TLatex;
  lat->SetNDC();
  lat->SetTextColor(2);
  lat->DrawLatex(0.35, 0.4, "expected (best-fit)");
  lat->SetTextColor(1);
  lat->DrawLatex(0.35, 0.33, "observed");

  // Call again aeffEestZd->CalcGammaEestDistribution with the final spectrum, this time to fill the expected gammas histogram and to find out what
  // fraction of events have true energy above or below (whatever is larger) the limits of each Eest bin. If that fraction is large, it makes no sense
  // to display a spectral point at that energy, since the number of events will be dominated by other energies!
  //
  TH1D* hexpected_gammas = (TH1D*) SpectralLikelihood->GetHexpected_gammas()->Clone("hexpected_gammas");
  TH1D* hEventFractionBeyond20percent = (TH1D*)hexpected_gammas->Clone("hEventFractionBeyond20percent");
  hEventFractionBeyond20percent->SetTitle("Expected fraction of events with Etrue below 0.8* or above 1.2* their estimated energy");
  hEventFractionBeyond20percent->GetYaxis()->SetTitle("Fraction");
  hEventFractionBeyond20percent->Reset();

  aeffEestZd->CalcGammaEestDistribution(abs_spec, hexpected_gammas, hEventFractionBeyond20percent);

  if (!PropagateMCuncertainties)
    for (Int_t ibin = 1; ibin <= hexpected_gammas->GetNbinsX(); ibin++)
      hexpected_gammas->SetBinError(ibin, 0.);
  
  TCanvas& canv2b = disp->AddTab("Expected gammas");
  gWizard->WhiteBack(canv2b);
  canv2b.SetLogx();
  canv2b.SetLogy();
  canv2b.SetGridx();
  canv2b.SetGridy();
  canv2b.SetLeftMargin(0.15);
  canv2b.SetBottomMargin(0.15);
  hexpected_gammas->SetStats(0);
  hexpected_gammas->GetXaxis()->SetRange(firstEbin, lastEbin);
  hexpected_gammas->SetMinimum(0.1); // If predicted # of gammas is very small, bin is not relevant...
  hexpected_gammas->DrawCopy("e");


  TCanvas& canv2c = disp->AddTab("Residuals");
  gWizard->WhiteBack(canv2c);
  canv2c.Divide(1,2);
  canv2c.cd(1);
  gPad->SetLogx();
  gPad->SetGridx();
  gPad->SetGridy();
  
  
  TH1D* chi2ContributionGauss = SpectralLikelihood->GetHchi2ContributionGauss();
  TH1D* chi2ContributionPoisson = SpectralLikelihood->GetHchi2ContributionPoisson();
  
  TH1D* hresidual = (TH1D*)hobserved_events->Clone("hresidual");
  hresidual->SetTitle("Fit residuals, relative observed-expected On events");
  hresidual->SetDirectory(0);
  hresidual->Divide(hexpected_events);  // Now it contains the ratio of observed to expected events
  hresidual->GetXaxis()->SetTitle("estimated energy (GeV)");
  hresidual->GetYaxis()->SetTitle("ON region, (observed-expected) / expected");
  hresidual->SetStats(0);

  TH1D* hresidual2 = (TH1D*)hobserved_events->Clone("hresidual2");
  hresidual2->SetTitle("Fit residuals, relative observed excess - expected gammas");
  hresidual2->SetDirectory(0);
  hresidual2->Add(hexpected_background, -1.); // Now it contains the excess events
  hresidual2->Divide(hexpected_gammas); // Now it contains the ratio of observed excess to expected (fitted) gammas


  hresidual2->GetXaxis()->SetTitle("estimated energy (GeV)");
  hresidual2->GetYaxis()->SetTitle("(observed excess-expected gammas) / expected gammas");
  hresidual2->SetStats(0);


  // AM 20141021: NOTE!!! We replace ROOT's calculation of the ratio histogram error bars (which assumes gaussianity, and also independence of the two values!) by the 
  // sqrt(Chi2) from likelihood calculation, which should be a better estimate of the deviation. It will make the residuals (w.r.t. 1) reflect better the actual 
  // deviation between model and data

  for (Int_t ibin = 1; ibin <= hresidual->GetNbinsX(); ibin++)
    {
      Double_t ratio = hresidual->GetBinContent(ibin);
      if (chi2ContributionGauss->GetBinContent(ibin) > 0.)
  	hresidual->SetBinError(ibin, fabs(ratio-1.)/sqrt(chi2ContributionGauss->GetBinContent(ibin)));
      else if (chi2ContributionPoisson->GetBinContent(ibin) > 0.)
  	hresidual->SetBinError(ibin, fabs(ratio-1.)/sqrt(chi2ContributionPoisson->GetBinContent(ibin)));

      ratio = hresidual2->GetBinContent(ibin);
      if (chi2ContributionGauss->GetBinContent(ibin) > 0.)
  	hresidual2->SetBinError(ibin, fabs(ratio-1.)/sqrt(chi2ContributionGauss->GetBinContent(ibin)));
      else if (chi2ContributionPoisson->GetBinContent(ibin) > 0.)
  	hresidual2->SetBinError(ibin, fabs(ratio-1.)/sqrt(chi2ContributionPoisson->GetBinContent(ibin)));
    }

  for (Int_t ibin = firstEbin; ibin <= lastEbin; ibin++)
    {
      hresidual->SetBinContent(ibin, hresidual->GetBinContent(ibin)-1.);   // Now it contains (observed-expected) / expected  (Non)
      hresidual2->SetBinContent(ibin, hresidual2->GetBinContent(ibin)-1.); // Now it contains (observed-expected) / expected  (excess)
    }

  hresidual->GetYaxis()->SetTitleSize(0.04);
  hresidual->GetYaxis()->SetLabelSize(0.05);
  hresidual->GetYaxis()->SetTitleOffset(0.7);

  gPad->SetBottomMargin(0.15);
  hresidual->GetXaxis()->SetRangeUser(20., 1.e5);
  hresidual->DrawCopy();
  lin->DrawLine(hresidual->GetXaxis()->GetBinLowEdge(firstEbin), hresidual->GetMinimum(), hresidual->GetXaxis()->GetBinLowEdge(firstEbin), hresidual->GetMaximum());
  lin->DrawLine(hresidual->GetXaxis()->GetBinUpEdge(lastEbin),   hresidual->GetMinimum(), hresidual->GetXaxis()->GetBinUpEdge(lastEbin),   hresidual->GetMaximum());
  
  canv2c.cd(2);
  gPad->SetLogx();
  gPad->SetGridx();
  gPad->SetGridy();
  gPad->SetBottomMargin(0.15);
  
  chi2ContributionGauss->SetLineColor(2);
  if (chi2ContributionPoisson->GetMaximum() > chi2ContributionGauss->GetMaximum())
    chi2ContributionGauss->SetMaximum(chi2ContributionPoisson->GetMaximum()*1.1);
  
  chi2ContributionGauss->GetYaxis()->SetTitle("Chi2");
  chi2ContributionGauss->GetYaxis()->SetTitleOffset(0.7);
  chi2ContributionGauss->GetXaxis()->SetRangeUser(20., 1.e5);
  chi2ContributionGauss->DrawCopy("hist");
  chi2ContributionPoisson->SetLineColor(4);
  chi2ContributionPoisson->DrawCopy("hist,same");
  TLegend* legchi = new TLegend(0.15,0.75,0.4,0.89);
  legchi->AddEntry(chi2ContributionPoisson, "Poissonian regime", "l");
  legchi->AddEntry(chi2ContributionGauss, "Gaussian regime", "l");
  legchi->Draw();


  TCanvas& canv2d = disp->AddTab("Residuals2");
  gWizard->WhiteBack(canv2d);
  canv2d.SetLogx();
  canv2d.SetGridx();
  canv2d.SetGridy();
  hresidual2->GetXaxis()->SetRangeUser(20., 1.e5);
  hresidual2->GetYaxis()->SetRangeUser(-5., 5.);
  hresidual2->DrawCopy();

  TCanvas& canv3 = disp->AddTab("dF/dE");
  gWizard->WhiteBack(canv3);
  canv3.SetLogx();
  canv3.SetLogy();
  canv3.SetGridx();
  canv3.SetGridy();

  // Since the averaged coll. area AeffEest (= <A'eff> , see below) of each bin depends on the spectrum, and we have obtained a new one,
  // we recalculated the AeffEest above. Hence we must re-calculate also the spectral points to be shown =>
  //
  // => modify (most often slightly) the spectral points from flute, to account for the fact that they were calculated (via Aeff vs. Eest) with
  // a tentative spectrum - they will now be re-calculated according to the new spectrum:
  //

  TGraphAsymmErrors* newDiffEspectrum = SpectralLikelihood->GetRecalculatedDiffEspectrum(kTRUE, kFALSE);
  // 2nd argument = kFALSE indicates the spectral points must be set at their true energy (the median Etrue of the excess events)
  newDiffEspectrum->SetName("DiffEspectrum");

  TH1* hframe = canv3.DrawFrame(0.2*TMath::MinElement(newDiffEspectrum->GetN(), newDiffEspectrum->GetX()),
				0.2*TMath::MinElement(newDiffEspectrum->GetN(), newDiffEspectrum->GetY()),
				4*TMath::MaxElement(newDiffEspectrum->GetN(), newDiffEspectrum->GetX()),
				4.*TMath::MaxElement(newDiffEspectrum->GetN(), newDiffEspectrum->GetY()));
  hframe->GetXaxis()->SetTitle("E (GeV)");
  hframe->GetYaxis()->SetTitle("dF/dE (TeV^{-1} cm^{-2} s^{-1})");
  hframe->GetXaxis()->SetTitleOffset(1.2);
  hframe->GetYaxis()->SetTitleOffset(1.2);

  if (confint)
    {
      confint->SetFillStyle(3001);
      confint->SetFillColor(20);
      // SEE COMMENTS ABOVE ON THESE CONFIDENCE INTERVALS! THEY DO NOT CORRESPOND TO FOLD'S FIT! THAT IS WHY WE DO NOT SHOW THEM.
      //      confint->Draw("3");
    }
  

  // Note in any case that the spectral points are not *the result* of the program fold! The result is the spectral parameters, obtained from the
  // observables (distribution of events in Eest). The flux points "vs. Etrue" we will show are just approximations! You will not obtain the same
  // identical fit parameters by performing a fit to those points: note that the "fold" code uses also Eest bins which may even lack a significant excess
  // with which to obtain a "flux point"!
  //

  newDiffEspectrum->SetTitle("Differential energy spectrum");
  newDiffEspectrum->SetMarkerStyle(20);
  newDiffEspectrum->DrawClone("p");
  
  TF1* spec = kIs_log_dFdE?
    new TF1("spec", Form("exp(%s)", func->GetExpFormula().Data()), 
	    func->GetXmin(), func->GetXmax()) :
    new TF1("spec", func->GetExpFormula().Data(), 
	    func->GetXmin(), func->GetXmax());
  spec->SetParameters(func->GetParameters());
  spec->SetLineStyle(2);
  spec->DrawClone("same");

  abs_spec->DrawClone("same"); // Clone is important, otherwise crash on exit because AbsorbedSpectrum is not available!

  TCanvas& canv4 = disp->AddTab("SED");
  gWizard->WhiteBack(canv4);
  canv4.SetLogx();
  canv4.SetLogy();
  canv4.SetGridx();
  canv4.SetGridy();
  gPad->SetMargin(0.1, 0.05, 0.15, 0.1);

  TF1* fsed = kIs_log_dFdE?
    new TF1("fsed", Form("1.e-6*x*x*exp(%s)", func->GetExpFormula().Data()), 
	    func->GetXmin(), func->GetXmax()) :
    new TF1("fsed", Form("1.e-6*x*x*%s", func->GetExpFormula().Data()), 
	    func->GetXmin(), func->GetXmax());
  fsed->SetParameters(func->GetParameters());

  fsed->SetLineStyle(2);

  TGraphAsymmErrors* g_sed = SpectralLikelihood->GetRecalculatedSED();
  g_sed->SetName("observed_sed");

  Double_t minPlottedE = 0.2*TMath::MinElement(g_sed->GetN(), g_sed->GetX());
  Double_t maxPlottedSED = 8.*TMath::MaxElement(g_sed->GetN(), g_sed->GetY());

  if (FermiE > 0. && FermidFdE > 0. && FermiIndex > 0.)
    {
      minPlottedE = FermiE / 10.;
      maxPlottedSED = TMath::Max(8.*fsed->Eval(FermiE), maxPlottedSED);
      maxPlottedSED = TMath::Max(8.*FermidFdE*pow(FermiE*1.e-3, 2.), maxPlottedSED);
    }

  hframe = canv4.DrawFrame(minPlottedE,
			   0.2*TMath::MinElement(g_sed->GetN(), g_sed->GetY()),
			   4.*TMath::MaxElement(g_sed->GetN(), g_sed->GetX()),
			   maxPlottedSED);

  hframe->GetXaxis()->SetTitle("E (GeV)");
  hframe->GetYaxis()->SetTitle("E^{2} dF/dE (TeV cm^{-2} s^{-1})");
  hframe->GetXaxis()->SetTitleOffset(1.2);
  hframe->GetYaxis()->SetTitleOffset(1.2);

  // Crab Nebula SED from MAGIC stereo, arXiv:1406.6892 :
  TF1* CrabSED = new TF1("CrabSED","x*x*1.e-6*3.23e-11*pow(x/1000.,-2.47-0.24*log10(x/1000.))",40,1.e5); // TeV cm-2 s-1
  CrabSED->SetLineColor(4);
  CrabSED->SetLineStyle(3);
  CrabSED->Draw("same");

  if (confintSED)
    {
      confintSED->SetFillStyle(3001);
      confintSED->SetFillColor(20);
      // SEE COMMENTS ABOVE ON THESE CONFIDENCE INTERVALS! THEY DO NOT CORRESPOND TO FOLD'S FIT! THAT IS WHY WE DO NOT SHOW THEM.
      //      confintSED->Draw("3");  // Just for occasional tests!
    }
  
  g_sed->SetMarkerStyle(20);
  g_sed->DrawClone("p");
  
  // Plot the absorbed (best-fit) SED:
  TF1* abs_sed = SpectralLikelihood->GetAbsorbedSED();

  abs_sed->DrawClone("same");

  // Now plot the de-absorbed SED with the assumed tau scaling factor:
  TGraphAsymmErrors* deabsorbed_sed = (TGraphAsymmErrors*)g_sed->Clone("deabsorbed_sed");
  Double_t *xsed = deabsorbed_sed->GetX(); 
  Double_t *ysed = deabsorbed_sed->GetY(); 
  Double_t *eyhsed = deabsorbed_sed->GetEYhigh();
  Double_t *eylsed = deabsorbed_sed->GetEYlow();

  for (Int_t ised = 0; ised < deabsorbed_sed->GetN(); ised++)
    {
      Double_t newy = TMath::Exp(tauscale*tauvsE->Eval(log10(xsed[ised])-3.)) * ysed[ised];
      deabsorbed_sed->SetPointError(ised, 0., 0., newy*eylsed[ised]/ysed[ised], newy*eyhsed[ised]/ysed[ised]);
      deabsorbed_sed->SetPoint(ised, xsed[ised], newy);
    }

  deabsorbed_sed->SetMarkerStyle(24);
  TLegend* leg = new TLegend(0.13,0.84,0.35,0.89);
  leg->AddEntry( deabsorbed_sed->DrawClone("p"), "Deabsorbed SED", "p");
  leg->Draw();
  
  TLegend* leg2 = new TLegend(0.45,0.84,0.94,0.89);
  leg2->SetTextSize(0.025);
  leg2->AddEntry(CrabSED, "Crab Nebula (MAGIC stereo, arXiv:1406.6892)", "l");
  leg2->Draw();
  
  hframe->SetMaximum(5*TMath::MaxElement(deabsorbed_sed->GetN(), deabsorbed_sed->GetY()));

  fsed->DrawClone("same");	  


  // Now Draw the Fermi butterfly, if Fermi data were used in the fit:

  if (FermiE > 0. && FermidFdE > 0. && FermiIndex > 0.)
	{
	  Double_t norm  = pow(FermiE*1.e-3, 2.)*FermidFdE;
	  Double_t dnorm = pow(FermiE*1.e-3, 2.)*FermiDeltadFdE;

	  TF1* fermi_hard_a = new TF1("fermi_hard_a", Form("[0]*pow(x/%f,-[1]+2)", FermiE),
				     FermiE/3., FermiE);
	  TF1* fermi_hard_b = new TF1("fermi_hard_b", Form("[0]*pow(x/%f,-[1]+2)", FermiE),
				     FermiE, 3.*FermiE);

	  fermi_hard_a->SetParameters(norm-dnorm, FermiIndex-FermiDeltaIndex);
	  fermi_hard_b->SetParameters(norm+dnorm, FermiIndex-FermiDeltaIndex);

	  TF1* fermi_soft_a = new TF1("fermi_soft_a", Form("[0]*pow(x/%f,-[1]+2)", FermiE),
				    FermiE/3., FermiE);
	  TF1* fermi_soft_b = new TF1("fermi_soft_b_%02d", Form("[0]*pow(x/%f,-[1]+2)", FermiE),
				    FermiE, 3.*FermiE);

	  fermi_soft_a->SetParameters(norm+dnorm, FermiIndex+FermiDeltaIndex);
	  fermi_soft_b->SetParameters(norm-dnorm, FermiIndex+FermiDeltaIndex);

	  fermi_hard_a->SetLineColor(4);
	  fermi_soft_a->SetLineColor(4);
	  fermi_hard_a->Draw("same");
	  fermi_soft_a->Draw("same");
	  fermi_hard_b->SetLineColor(4);
	  fermi_soft_b->SetLineColor(4);
	  fermi_hard_b->Draw("same");
	  fermi_soft_b->Draw("same");
	}

  TCanvas& canv5 = disp->AddTab("Spillover");
  gWizard->WhiteBack(canv5);
  canv5.SetLogx();
  canv5.SetGridx();
  canv5.SetGridy();
  hEventFractionBeyond20percent->SetStats(0);
  hEventFractionBeyond20percent->DrawCopy();

  if (WriteOutput)
    {
      MMarsVersion* marsversion_flute = 0;
      TFile finput(inputfile);
      if (finput.FindKey("MMarsVersion_flute"))
	{
	  marsversion_flute = new MMarsVersion("MMarsVersion_flute");
	  marsversion_flute->Read("MMarsVersion_flute");
	}
      finput.Close();

      TFile* outputfile;
      outputfile = new TFile("Output_fold.root", "recreate");

      if (marsversion_flute)
	marsversion_flute->Write();
      MMarsVersion marsversion_fold("MMarsVersion_fold");
      marsversion_fold.Write();

      TParameter<float> etime;
      etime.SetVal(efftime);
      etime.Write("efftime");
  
      TParameter<float> eblscale;
      if (nparsEBL > 0)
	eblscale.SetVal(xs[npars-1]);
      else
	eblscale.SetVal(scaleOpticalDepth);
      eblscale.Write("eblscale");
      
      if (nparsEBL > 0)
	{
	  TParameter<float> err_eblscale_up;
	  TParameter<float> err_eblscale_low;
	  err_eblscale_up.SetVal(errup);
	  err_eblscale_low.SetVal(errlow);
	  err_eblscale_up.Write("ebl_errup");
	  err_eblscale_low.Write("ebl_errlow");
	}

    //Store the fit results as TVector object in Output_fold.root

      //Store parameters results
      TVector par(nparsIntrinsic);
      for (Int_t ipar = 0; ipar <= nparsIntrinsic-1; ipar++){

        par[ipar] = func->GetParameter(ipar);
        
      }
      par.Write("FitParameters");

      //Store parameters MIGRAD uncertainties 
      TVector migrad_par_error(nparsIntrinsic);
      for (Int_t ipar = 0; ipar <= nparsIntrinsic-1; ipar++){

        migrad_par_error[ipar] = delta_xs[ipar];
        
      }
      migrad_par_error.Write("MIGRAD_FitParameters_errors");

      corrmatrix.Write("CorrelationMatrix");

      //Store parameters MINOS uncertainties 
      TVector minos_par_errorHigh(nparsIntrinsic);
      TVector minos_par_errorLow(nparsIntrinsic);
      for (Int_t ipar = 0; ipar <= nparsIntrinsic-1; ipar++){
        
        minos_par_errorHigh[ipar] = deltaparhigh[ipar];
        minos_par_errorLow[ipar] = deltaparlow[ipar];

      }
      minos_par_errorHigh.Write("MINOS_FitParameters_errorsHigh");
      minos_par_errorLow.Write("MINOS_FitParameters_errorsLow");


      TParameter<int> numdof;
      numdof.SetVal(ndof);
      numdof.Write("ndof");
      
      TParameter<float> chi2val;
      chi2val.SetVal(chisquare);
      chi2val.Write("chisquare");
      
      TParameter<float> fitprob;
      fitprob.SetVal(fprobability);
      fitprob.Write("fitprob");
      
      hobserved_events->Write("hobserved_events");
      
      hexpected_gammas->SetLineColor(2);
      hexpected_gammas->Write("hexpected_gammas");

      hexpected_background->SetLineColor(2);
      hexpected_background->Write("hexpected_background");
      
      hEstBckgE->Write("hEstBckgE");

      fsed->Write();
      spec->Write();
      abs_sed->Write("abs_sed");
      abs_spec->Write("abs_spec");
      g_sed->Write();
      deabsorbed_sed->Write();
      hresidual->Write();
      chi2ContributionPoisson->Write("chi2ContributionPoisson");
      chi2ContributionGauss->Write("chi2ContributionGauss");
      func->Write("SpectralModel");

      // NOTE!! We do not write out these confidence intervals because they DO NOT correspond
      // to our maximum likelihood fit, but to the preliminary ROOT fits we did to get startingparameters.
      //
      // if (confint)
      // 	confint->Write();
      // if (confintSED)
      // 	confintSED->Write();
      
      hEventFractionBeyond20percent->Write();
      
      // AM: I have no idea why the writing of the MHMcCollectionArea object makes the program crash

      if (SpectralLikelihood->GetAeffEestZd())
	{
	  // in some occasions, while in others it works fine. For now I am forced
	  // to disable this...	
	  //	SpectralLikelihood->GetAeffEestZd()->Write();
	  SpectralLikelihood->GetAeffEestZd()->GetHistCoarse()->ProjectionX("AeffEest")->Write();
	  outputfile->mkdir("ExposureVsEtrue");
	  outputfile->cd("ExposureVsEtrue");
	  for (Int_t ibin = SpectralLikelihood->GetFirstBin();
	       ibin <= SpectralLikelihood->GetLastBin(); ibin++)
	    SpectralLikelihood->GetAeffEestZd()->GetHistExposureInEestBin(ibin)->Write();
	}
	  
      if (SpectralLikelihood->GetHDeltaEtot_eest())
	SpectralLikelihood->GetHDeltaEtot_eest()->Write();
      

      outputfile->Close();
      delete outputfile;
    }

  
  if (chi2timer)
    gLog << inf << "Real time used by chi2 function: " << chi2timer->RealTime() << " seconds." << endl;


  if (WriteOutput)
    disp->SaveAsRoot("Status_fold.root");

  // Clean-up:
  delete func;
  delete newDiffEspectrum;
  delete hresidual;
  delete g_sed;
  delete fsed;
  delete deabsorbed_sed;

  if (deletedisp)
    {
      delete disp;
      delete legchi;
      delete lin;
      delete lat;
      delete leg;
      if (confint)
	delete confint;
      if (confintSED)
	delete confintSED;
    }

  delete SpectralLikelihood;

  gLog << inf << endl << "Decorrelation energy: " << Form("%.2f", decorrelE) << " GeV. Used normalization energy: " 
       << Form("%.2f", normE) << " GeV." << endl;
  if (fabs(decorrelE-normE) > 0.1)  // NOT an error, I write it in red and at the end so users do not overlook it:
    gLog << err << "NOTE: Use --NormalizationE=" << Form("%.2f", decorrelE) << " if you want flux normalization at the decorrelation energy (recommended)" << endl;

  gLog << inf << endl << "fold finished successfully!" << endl << endl;

  
  return chisquare;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Note: Enorm (in GeV) is the normalization energy (not a free parameter!)
//
Bool_t createFunction(TString funcname, TF1* fn, Double_t Enorm, Double_t* x, Double_t* step)
{
  TF1* func;
  
  Double_t minE = 0.1; // GeV
  Double_t maxE = 1.e5; // GeV
  

  if (funcname == TString("PWL"))
    {
      // Flux per cm-2 TeV-1 s-1    power-law:
      func = new TF1(funcname, Form("[0]*pow(x/%.2f,[1])", Enorm), minE, maxE);
      x[0] = 1.e-9; x[1] = -2.;

      func->SetParameters(x[0], x[1]);
      step[0] = 1.0e-12; step[1] = 1.0;
    }
  else if (funcname == TString("LP"))
    {
      // Log parabola with negative curvature  (forbid convex spectra!):
      func = new TF1(funcname, Form("[0]*pow(x/%.2f,[1]-[2]*[2]*log10(x/%.2f))", Enorm, Enorm), minE, maxE);
      x[0] = 1.e-9; x[1] = -2.; x[2] = 0.15;

      func->SetParameters(x[0], x[1], x[2]);
      step[0] = 1.0e-12; step[1] = 1.0; step[2] = 0.1;
    }
  else if (funcname == TString("freeLP"))
    {
      // Log parabola with negative or positive curvature (i.e. convex or concave):
      func = new TF1(funcname, Form("[0]*pow(x/%.2f,[1]-[2]*log10(x/%.2f))", Enorm, Enorm), minE, maxE);
      x[0] = 1.e-9; x[1] = -2.; x[2] = 0.;

      func->SetParameters(x[0], x[1], x[2]);
      step[0] = 1.0e-12; step[1] = 1.0; step[2] = 0.1;
    }
  else if (funcname == TString("ELP"))
    {
      // logparabola with exp cut-off  (forbid convex spectra!):
      func = new TF1(funcname, Form("[0]*pow(x/%.2f,[1]-[2]*[2]*log10(x/%.2f))*exp(-x/[3]/[3])", Enorm, Enorm), minE, maxE);
      x[0] = 1.e-9; x[1] = -2.; x[2] = 0.15; x[3] = sqrt(3.e3);
      func->SetParameters(x[0], x[1], x[2], x[3]);
      step[0] = 1.0e-12; step[1] = 1.0; step[2] = 0.1; step[3] = sqrt(500.);
    }
  else if (funcname == TString("EPWL"))
    {
      // power-law with exp cut-off  (forbid convex spectra!):
      func = new TF1(funcname, Form("[0]*pow(x/%.2f,[1])*exp(-x/[2]/[2])", Enorm), minE, maxE);
      x[0] = 1.e-9; x[1] = -2.; x[2] = sqrt(3.e3);
      func->SetParameters(x[0], x[1], x[2]);
      step[0] = 1.0e-12; step[1] = 1.0; step[2] = sqrt(500.);
    }
  else if (funcname == TString("freeEPWL"))
    {
      // power-law with exp cut-off (i.e. convex or concave):
      func = new TF1(funcname, Form("[0]*pow(x/%.2f,[1])*exp(-x/[2])", Enorm), minE, maxE);
      x[0] = 1.e-9; x[1] = -2.; x[2] = sqrt(3.e6);
      func->SetParameters(x[0], x[1], x[2]);
      step[0] = 1.0e-12; step[1] = 1.0; step[2] = sqrt(500.);
    }
  else if (funcname == TString("SEPWL"))
    {
      // power-law with super-exp cut-off (forbid convex spectra!):
      func = new TF1(funcname, Form("[0]*pow(x/%.2f,[1])*exp(-pow(x/[2]/[2],[3]))", Enorm), minE, maxE);
      x[0] = 1.e-9; x[1] = -2.; x[2] = sqrt(3.e3); x[3] = 1.;
      func->SetParameters(x[0], x[1], x[2], x[3]);
      step[0] = 1.0e-12; step[1] = 1.0; step[2] = sqrt(500.); step[3] = 1.;
    }
  else if (funcname == TString("CMBPWL"))
    { 
      Double_t minE2 = 60.;
      Double_t maxE2 = 4.e3;
      // Concave Multiply broken power law with 5 knots:
      Float_t breaks[4]; //change 4 to knots-1
      Double_t deltalogE = (log (maxE2) - log (minE2))/ 4; //change /4 to /knots-1
      for (int i = 0; i <= 3; i = i + 1){ //change 3 to knots-2
        breaks[i] = exp (log(minE2) + i * deltalogE);
      }
      Float_t break0 = breaks[0];
      Float_t break1 = breaks[1];
      Float_t break2 = breaks[2];
      Float_t break3 = breaks[3]; // add break3, break4,... until break(knots-2)

      func = new TF1(funcname, Form("(x<%f? [0] * pow(x/%f,[1]) : (x<%f? [0] * pow(%f/%f,[1]) * pow(x/%f,([1] - pow([2],2))) : (x<%f? [0] * pow(%f/%f,[1]) * pow(%f/%f,([1] - pow([2],2))) * pow(x/%f,([1] - pow([2],2) - pow([3],2))) : [0] * pow(%f/%f,[1]) * pow(%f/%f,([1] - pow([2],2))) * pow(%f/%f,([1] - pow([2],2) - pow([3],2))) * pow(x/%f,([1] - pow([2],2) - pow([3],2) - pow([4],2))))))", //add the equation of the next power laws untill you have the needed number of power laws.
			break1, break0,
			break2, break1, break0, break1,
			break3, break1, break0, break2, break1, break2,
      break1, break0, break2, break1, break3, break2, break3), //add one row following the same pattern 
		  minE, maxE);
      x[0] = 1.e-9; x[1] = -1.; x[2] = 0.7; x[3]= 0.7; x[4] = 0.7;
      func->SetParameters(x[0], x[1], x[2], x[3], x[4]);
      step[0] = 1.0e-11; step[1] = 0.05; step[2] = 0.05; step[3] = 0.05; step[4]= 0.05;
    }
  //
  // NOTE, AM: the smothly broken power-law is very problematic convergence-wise: fails all the time, even for data which should
  // fit well a SBWL, unless the range is set to keep only significant points, and good starting parameters are set.
  //
  // else if (funcname == TString("SBPWL"))
  //   {
  //     func = new TF1(funcname, Form("[0]*pow(x/%f,[1])*pow(1.+pow(x/[2],([1]-[3])/[4]),-[4])", Enorm), minE, maxE);  // no concavity constraint
  //     // [1]: low-E index,   [2]: break ,  [3]: high-E index, [4]: break smoothness (the larger the smoother)      
  //     x[0] = 1.e-9; x[1] = -2.; x[2] = 300.; x[3] = -3.; x[4] = 0.1;
  //     func->SetParameters(x[0], x[1], x[2], x[3], x[4]);

  //     step[0] = 1.0e-12; step[1] = 0.1; step[2] = 10.; step[3] = 0.1; step[4] = 0.01;
  //   }
  else
    return kFALSE;

  func->Copy(*fn);

  delete func;
  fn->SetName(funcname);  // Important! The function will later be used by its name.
  
  return kTRUE;

}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Note: Enorm (in GeV) is the normalization energy (not a free parameter!)
// This creates functions which return (natural) logarithms of fluxes!
//
Bool_t createFunctionLog(TString funcname, TF1* fn, Double_t Enorm, Double_t* x, Double_t* step)
{
  TF1* func;

  Double_t minE = 0.1; // GeV
  Double_t maxE = 1.e5; // GeV

  if (funcname == TString("PWL"))
    {
      // Flux per cm-2 TeV-1 s-1    power-law:
      func = new TF1(funcname, Form("[0]+[1]*log(x/%.2f)", Enorm), minE, maxE);
      x[0] = log(1.e-9); x[1] = -2.;

      func->SetParameters(x[0], x[1]);
      step[0] = 1.0; step[1] = 1.0;
    }
  else if (funcname == TString("LP"))
    {
      // Log parabola with negative curvature  (forbid convex spectra!):
      func = new TF1(funcname, Form("[0]+([1]-[2]*[2]*log10(x/%.2f))*log(x/%.2f)", Enorm, Enorm), minE, maxE);
      x[0] = log(1.e-9); x[1] = -2.; x[2] = 0.15;

      func->SetParameters(x[0], x[1], x[2]);
      step[0] = 1.0; step[1] = 1.0; step[2] = 0.1;
    }
//  else if (funcname == TString("freeLP"))  // no concaveness requirement.
//    {
//      // Log parabola with negative or positive curvature (i.e. convex or concave)
//      // We choose parameters such that one of them, [1], is the (negative) photon index at 300 GeV. We do this so that we can later
//      // set a limit on it, i.e. not harder than "^-1.5
//      func = new TF1(funcname, Form("[0]+[1]*log(x/%.2f)+[2]*(log(x/%.2f)*log(x/%.2f)-2*log(300./%.2f)*log(x/%.2f))",
//				    Enorm, Enorm, Enorm, Enorm, Enorm), minE, maxE);
//
//      x[0] = log(1.e-9); x[1] = -2.; x[2] = 0.;
//      func->SetParameters(x[0], x[1], x[2]);
//      step[0] = 1.0; step[1] = 0.1; step[2] = 0.1;
//    }
  else if (funcname == TString("freeLP"))  // no concaveness requirement
    {
      // Log parabola with negative or positive curvature (i.e. convex or concave):
      func = new TF1(funcname, Form("[0]+([1]-[2]*log10(x/%.2f))*log(x/%.2f)", Enorm, Enorm), minE, maxE);
      x[0] = log(1.e-9); x[1] = -2.; x[2] = 0.;
      func->SetParameters(x[0], x[1], x[2]);
      step[0] = 1.0; step[1] = 1.0; step[2] = 0.1;
    }
  else if (funcname == TString("ELP"))
    {
      // logparabola with exp cut-off  (forbid convex spectra!):
      func = new TF1(funcname, Form("[0]+([1]-[2]*[2]*log10(x/%.2f))*log(x/%.2f)-x/[3]/[3]", Enorm, Enorm), minE, maxE);
      x[0] = log(1.e-9); x[1] = -2.; x[2] = 0.15; x[3] = sqrt(3.e4);
      func->SetParameters(x[0], x[1], x[2], x[3]);
      step[0] = 1.0; step[1] = 1.0; step[2] = 0.1; step[3] = sqrt(500.);
    }
  else if (funcname == TString("EPWL"))
    {
      // power-law with exp cut-off  (forbid convex spectra!):
      func = new TF1(funcname, Form("[0]+[1]*log(x/%.2f)-x/[2]/[2]", Enorm), minE, maxE);
      // TEST::
      //            Enorm = 1.e4;   // 10 TeV
      //            func = new TF1(funcname, Form("[0]+%.2f/[2]/[2]+[1]*log(x/%.2f)-x/[2]/[2]", Enorm, Enorm), minE, maxE);

      x[0] = log(1.e-9); x[1] = -2.; x[2] = sqrt(3.e4);
      func->SetParameters(x[0], x[1], x[2]);
      step[0] = 1.0; step[1] = 1.0; step[2] = sqrt(500.);
    }
  else if (funcname == TString("freeEPWL"))  // no concaveness requirement
    {
      // power-law with exp cut-off (i.e. convex or concave):
      func = new TF1(funcname, Form("[0]+[1]*log(x/%.2f)-x/[2]", Enorm), minE, maxE);
      // TEST::
      //            Enorm = 1.e4;   // 10 TeV
      //            func = new TF1(funcname, Form("[0]+%.2f/[2]/[2]+[1]*log(x/%.2f)-x/[2]/[2]", Enorm, Enorm), minE, maxE);

      x[0] = log(1.e-9); x[1] = -2.; x[2] = sqrt(3.e8);
      func->SetParameters(x[0], x[1], x[2]);
      step[0] = 1.0; step[1] = 1.0; step[2] = sqrt(500.);
    }
  else if (funcname == TString("pileupEPWL"))
    {
      // power-law with exp pile-up:
      func = new TF1(funcname, Form("[0]+[1]*log(x/%.2f)+x/[2]", Enorm), minE, maxE);
      x[0] = log(1.e-9); x[1] = -2.; x[2] = 3.e2;
      func->SetParameters(x[0], x[1], x[2]);
      step[0] = 1.0; step[1] = 1.0; step[2] = 50.;
    }
  else if (funcname == TString("SEPWL"))
    {
      // power-law with super-exp cut-off (forbid convex spectra!):
      func = new TF1(funcname, Form("[0]+[1]*log(x/%.2f)-pow(x/[2]/[2],[3])", Enorm), minE, maxE);
      x[0] = log(1.e-9); x[1] = -2.; x[2] = sqrt(3.e4); x[3] = 1.;
      func->SetParameters(x[0], x[1], x[2], x[3]);
      step[0] = 1.0; step[1] = 1.0; step[2] = sqrt(500.); step[3] = 0.1;
    }
  else if (funcname == TString("SELP"))
    {
      // logparabola with super-exponential cut-off  (forbid convex spectra!):
      func = new TF1(funcname, Form("[0]+([1]-[2]*[2]*log10(x/%.2f))*log(x/%.2f)-pow(x/[3]/[3],[4])", Enorm, Enorm), minE, maxE);
      x[0] = log(1.e-9); x[1] = -2.; x[2] = 0.15; x[3] = sqrt(3.e4); x[4] = 1.;
      func->SetParameters(x[0], x[1], x[2], x[3], x[4]);
      step[0] = 1.0; step[1] = 1.0; step[2] = 0.1; step[3] = sqrt(500.); step[4] = 0.1;
    }
  else
    return kFALSE;

  func->Copy(*fn);

  delete func;
  fn->SetName(funcname);  // Important! The function will later be used by its name.

  return kTRUE;

}

