
/*
Copyright 2005 University of Warwick

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
Laura++ package authors:
John Back
Paul Harrison
Thomas Latham
*/

// This example reproduces the CP-violating isobar model used by the LHCb
// amplitude analysis of B+- -> pi+ pi+ pi- (Phys. Rev. D 101, 012006).
// Each isobar component is given an independent Cartesian CP coefficient,
// following the convention of the paper's Table XXI:
//   c_q = (x + q*dx) + i*(y + q*dy),   q = +1 (B+), q = -1 (B-)
// which maps directly onto Laura++'s LauCartesianCPCoeffSet, whose
// particle/antiparticle coefficients are (x+dx, y+dy) and (x-dx, y-dy)
// respectively.

#include "LauCPFitModel.hh"
#include "LauCartesianCPCoeffSet.hh"
#include "LauDaughters.hh"
#include "LauEffModel.hh"
#include "LauIsobarDynamics.hh"
#include "LauRandom.hh"
#include "LauResonanceMaker.hh"
#include "LauVetoes.hh"

#include "TString.h"

#include <cstdlib>
#include <iostream>
#include <vector>

void usage( std::ostream& out, const TString& progName )
{
    out << "Usage:\n";
    out << progName << " gen [nExpt = 1] [firstExpt = 0] [seed = 65539+firstExpt]\n";
    out << "or\n";
    out << progName << " fit <iFit> [nExpt = 1] [firstExpt = 0]" << std::endl;
}

int main( int argc, char** argv )
{
    // Process command-line arguments
    // Usage:
    // ./GenFit3pi gen [nExpt = 1] [firstExpt = 0] [seed]
    // or
    // ./GenFit3pi fit <iFit> [nExpt = 1] [firstExpt = 0]
    //
    // The optional seed argument to "gen" lets you produce independent toy
    // samples (e.g. for an ensemble to be fitted externally): run once per
    // toy with a distinct firstExpt (used as the toy index in the output
    // file name) and/or a distinct seed. If no seed is given explicitly, one
    // is derived from firstExpt so that varying firstExpt alone is already
    // enough to get statistically independent samples.
    if ( argc < 2 ) {
        usage( std::cerr, argv[0] );
        return EXIT_FAILURE;
    }

    TString command = argv[1];
    command.ToLower();
    Int_t iFit( 0 );
    Int_t nExpt( 1 );
    Int_t firstExpt( 0 );
    Bool_t isToy( kTRUE );
    UInt_t seed( 0 );
    Bool_t haveSeed( kFALSE );
    if ( command == "gen" ) {
        if ( argc > 2 ) {
            nExpt = atoi( argv[2] );
            if ( argc > 3 ) {
                firstExpt = atoi( argv[3] );
                if ( argc > 4 ) {
                    seed     = static_cast<UInt_t>( atoi( argv[4] ) );
                    haveSeed = kTRUE;
                }
            }
        }
    } else if ( command == "fit" ) {
        if ( argc < 3 ) {
            usage( std::cerr, argv[0] );
            return EXIT_FAILURE;
        }
        iFit = atoi( argv[2] );
        if ( argc > 3 ) {
            nExpt = atoi( argv[3] );
            if ( argc > 4 ) {
                firstExpt = atoi( argv[4] );
            }
        }
    } else {
        usage( std::cerr, argv[0] );
        return EXIT_FAILURE;
    }

    // Laura++'s random generator otherwise defaults to a fixed seed (65539),
    // so separate process invocations would produce identical toys unless we
    // seed them differently here.
    if ( ! haveSeed ) {
        seed = 65539 + static_cast<UInt_t>( firstExpt );
    }
    LauRandom::setSeed( seed );

    // If you want to use square DP histograms for efficiency,
    // backgrounds or you just want the square DP co-ordinates
    // stored in the toy MC ntuple then set this to kTRUE
    Bool_t squareDP = kFALSE;

    // These define the DP => decay is B+ -> pi+ pi+ pi- and its charge conjugate
    // The DP is defined in terms of m13Sq and m23Sq
    LauDaughters* negDaughters = new LauDaughters( "B-", "pi-", "pi-", "pi+", squareDP );
    LauDaughters* posDaughters = new LauDaughters( "B+", "pi+", "pi+", "pi-", squareDP );

    // Optionally apply some vetoes to the DP
    // (example syntax given but commented-out)
    LauVetoes* negVetoes = new LauVetoes();
    LauVetoes* posVetoes = new LauVetoes();
    //Double_t DMin = 1.70;
    //Double_t DMax = 1.925;
    //negVetoes->addMassVeto(1, DMin, DMax); // D0 veto, m23 (and automatically m13)
    //posVetoes->addMassVeto(1, DMin, DMax); // D0 veto, m23 (and automatically m13)

    // Define the efficiency models (default to unity everywhere)
    LauEffModel* negEffModel = new LauEffModel( negDaughters, negVetoes );
    LauEffModel* posEffModel = new LauEffModel( posDaughters, posVetoes );

    // Set the values of the Blatt-Weisskopf barrier radii and whether they are fixed or floating.
    // The paper uses a common radius of 4.0 (GeV/c)^-1 for both the resonance and the parent.
    LauResonanceMaker& resMaker = LauResonanceMaker::get();
    resMaker.setDefaultBWRadius( LauBlattWeisskopfFactor::Parent, 4.0 );
    resMaker.setDefaultBWRadius( LauBlattWeisskopfFactor::Light, 4.0 );
    resMaker.fixBWRadius( LauBlattWeisskopfFactor::Parent, kTRUE );
    resMaker.fixBWRadius( LauBlattWeisskopfFactor::Light, kTRUE );

    // Create the isobar models for B- and B+.
    // resPairAmpInt arguments: resPairAmpInt = 1 => resonance mass is m23,
    // i.e. the invariant mass of the opposite-sign pion pair.
    LauIsobarDynamics* negSigModel = new LauIsobarDynamics( negDaughters, negEffModel );
    LauIsobarDynamics* posSigModel = new LauIsobarDynamics( posDaughters, posEffModel );

    LauAbsResonance* res( 0 );

    // rho(770)0: Gounaris-Sakurai lineshape, isobar-fit mass/width from Table XX.
    // This is used as the reference amplitude, so its x/y coefficients are fixed to (1,0).
    res = negSigModel->addResonance( "rho0(770)", 1, LauAbsResonance::GS );
    res->changeResonance( 0.7708, 0.1534, 1 );
    res = posSigModel->addResonance( "rho0(770)", 1, LauAbsResonance::GS );
    res->changeResonance( 0.7708, 0.1534, 1 );

    // omega(782): relativistic Breit-Wigner, nominal PDG/Laura++ mass and width.
    res = negSigModel->addResonance( "omega(782)", 1, LauAbsResonance::RelBW );
    res = posSigModel->addResonance( "omega(782)", 1, LauAbsResonance::RelBW );

    // f_2(1270): relativistic Breit-Wigner, isobar-fit mass from Table XX (nominal width).
    res = negSigModel->addResonance( "f_2(1270)", 1, LauAbsResonance::RelBW );
    res->changeResonance( 1.256, 0.1867, 2 );
    res = posSigModel->addResonance( "f_2(1270)", 1, LauAbsResonance::RelBW );
    res->changeResonance( 1.256, 0.1867, 2 );

    // rho(1450)0: relativistic Breit-Wigner, nominal PDG/Laura++ mass and width.
    res = negSigModel->addResonance( "rho0(1450)", 1, LauAbsResonance::RelBW );
    res = posSigModel->addResonance( "rho0(1450)", 1, LauAbsResonance::RelBW );

    // rho_3(1690)0: relativistic Breit-Wigner, nominal PDG/Laura++ mass and width.
    res = negSigModel->addResonance( "rho0_3(1690)", 1, LauAbsResonance::RelBW );
    res = posSigModel->addResonance( "rho0_3(1690)", 1, LauAbsResonance::RelBW );

    // Rescattering: KK-pipi inelastic scattering amplitude (LauRescatteringRes).
    res = negSigModel->addResonance( "Rescattering", 1, LauAbsResonance::Rescattering );
    res = posSigModel->addResonance( "Rescattering", 1, LauAbsResonance::Rescattering );

    // sigma / f_0(600): scalar pole lineshape, pole parameters from Table XXII.
    res = negSigModel->addResonance( "sigma0", 1, LauAbsResonance::Pole );
    res->changeResonance( 0.563, 0.350, 0 );
    res = posSigModel->addResonance( "sigma0", 1, LauAbsResonance::Pole );
    res->changeResonance( 0.563, 0.350, 0 );

    // Set the file names for the integrals information (can be useful for debugging)
    negSigModel->setIntFileName( "integ_neg_3pi.dat" );
    posSigModel->setIntFileName( "integ_pos_3pi.dat" );

    // Set the maximum signal DP ASq value.
    // If you do not provide a value, one will be determined automatically,
    // which should be close to the true maximum but is not guaranteed to
    // be optimal.
    // Any value, whether manually provided or automatically determined,
    // will be automatically adjusted to avoid bias or extreme inefficiency
    // but it is best to set this by hand once you've found the right value
    // through some trial and error.
    //negSigModel->setASqMaxValue( 3.0 );
    //posSigModel->setASqMaxValue( 3.0 );

    // Create the fit model, giving it both isobar models
    LauCPFitModel* fitModel = new LauCPFitModel( negSigModel, posSigModel );

    // Create the complex coefficients for the isobar model, using the Cartesian
    // CP convention: c_q = (x + q*dx) + i*(y + q*dy), q = +1 (B+), q = -1 (B-).
    // Values are the central Table XXI numbers from the paper.
    // rho(770)0 is the reference amplitude: x, y and dy are fixed, only dx floats.
    LauAbsCoeffSet* rho770Coeff = new LauCartesianCPCoeffSet( "rho0(770)",
                                                               1.000,
                                                               0.000,
                                                               -0.003,
                                                               0.000,
                                                               kTRUE,
                                                               kTRUE,
                                                               kFALSE,
                                                               kTRUE );
    LauAbsCoeffSet* omega782Coeff = new LauCartesianCPCoeffSet( "omega(782)",
                                                                 0.091,
                                                                 -0.007,
                                                                 0.000,
                                                                 -0.022,
                                                                 kFALSE,
                                                                 kFALSE,
                                                                 kFALSE,
                                                                 kFALSE );
    LauAbsCoeffSet* f2Coeff = new LauCartesianCPCoeffSet( "f_2(1270)",
                                                           0.291,
                                                           0.204,
                                                           -0.002,
                                                           -0.179,
                                                           kFALSE,
                                                           kFALSE,
                                                           kFALSE,
                                                           kFALSE );
    LauAbsCoeffSet* rho1450Coeff = new LauCartesianCPCoeffSet( "rho0(1450)",
                                                                -0.223,
                                                                0.191,
                                                                0.031,
                                                                0.068,
                                                                kFALSE,
                                                                kFALSE,
                                                                kFALSE,
                                                                kFALSE );
    LauAbsCoeffSet* rho3Coeff = new LauCartesianCPCoeffSet( "rho0_3(1690)",
                                                             0.073,
                                                             -0.045,
                                                             0.044,
                                                             -0.013,
                                                             kFALSE,
                                                             kFALSE,
                                                             kFALSE,
                                                             kFALSE );
    LauAbsCoeffSet* rescatteringCoeff = new LauCartesianCPCoeffSet( "Rescattering",
                                                                     0.142,
                                                                     -0.040,
                                                                     -0.047,
                                                                     -0.027,
                                                                     kFALSE,
                                                                     kFALSE,
                                                                     kFALSE,
                                                                     kFALSE );
    LauAbsCoeffSet* sigmaCoeff = new LauCartesianCPCoeffSet( "sigma0",
                                                              -0.485,
                                                              0.284,
                                                              0.231,
                                                              0.270,
                                                              kFALSE,
                                                              kFALSE,
                                                              kFALSE,
                                                              kFALSE );

    std::vector<LauAbsCoeffSet*> coeffset;
    coeffset.push_back( rho770Coeff );
    coeffset.push_back( omega782Coeff );
    coeffset.push_back( f2Coeff );
    coeffset.push_back( rho1450Coeff );
    coeffset.push_back( rho3Coeff );
    coeffset.push_back( rescatteringCoeff );
    coeffset.push_back( sigmaCoeff );
    for ( std::vector<LauAbsCoeffSet*>::iterator iter = coeffset.begin(); iter != coeffset.end();
          ++iter ) {
        fitModel->setAmpCoeffSet( *iter );
    }

    // Set the signal yield and define whether it is fixed or floated
    const Double_t nSigEvents = 50000.0;
    Bool_t fixNSigEvents      = kFALSE;
    LauParameter* signalEvents =
        new LauParameter( "signalEvents", nSigEvents, -1.0 * nSigEvents, 2.0 * nSigEvents, fixNSigEvents );
    fitModel->setNSigEvents( signalEvents );

    // Set the number of experiments to generate or fit and which
    // experiment to start with
    fitModel->setNExpts( nExpt, firstExpt, isToy );

    // Configure various fit options

    // Switch on/off calculation of asymmetric errors.
    fitModel->useAsymmFitErrors( kFALSE );

    // Randomise initial fit values for the signal mode
    fitModel->useRandomInitFitPars( kTRUE );

    const Bool_t haveBkgnds = ( fitModel->nBkgndClasses() > 0 );

    // Switch on/off Poissonian smearing of total number of events
    fitModel->doPoissonSmearing( haveBkgnds );

    // Switch on/off Extended ML Fit option
    fitModel->doEMLFit( haveBkgnds );

    // Generate toy from the fitted parameters
    //TString fitToyFileName("fitToyMC_3pi_");
    //fitToyFileName += iFit;
    //fitToyFileName += ".root";
    //fitModel->compareFitData(100, fitToyFileName);

    // Write out per-event likelihoods and sWeights
    //TString splotFileName("splot_3pi_");
    //splotFileName += iFit;
    //splotFileName += ".root";
    //fitModel->writeSPlotData(splotFileName, "splot", kFALSE);

    // Set the names of the files to read/write.
    // The toy index (firstExpt) is embedded in the generated file name so
    // that separate invocations (e.g. one per toy in an ensemble) don't
    // overwrite each other.
    TString dataFile( "gen-3pi_toy" );
    dataFile += firstExpt;
    dataFile += ".root";
    TString treeName( "genResults" );
    TString rootFileName( "" );
    TString tableFileName( "" );
    if ( command == "fit" ) {
        rootFileName   = "fit3pi_";
        rootFileName  += iFit;
        rootFileName  += "_expt_";
        rootFileName  += firstExpt;
        rootFileName  += "-";
        rootFileName  += ( firstExpt + nExpt - 1 );
        rootFileName  += ".root";
        tableFileName  = "fit3piResults_";
        tableFileName += iFit;
    } else {
        rootFileName  = "dummy.root";
        tableFileName = "gen3piResults";
    }

    // Execute the generation/fit
    fitModel->run( command, dataFile, treeName, rootFileName, tableFileName );

    return EXIT_SUCCESS;
}
