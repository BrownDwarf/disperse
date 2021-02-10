import os

import numpy as np
import pandas as pd
import os
import glob
from astropy.io import fits
import emcee
from astropy.time import Time
from tqdm import tqdm
from multiprocessing import Pool
import time
import pickle

import warnings

import pandas as pd
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

calcium_line = 8542


def get_goldilocks_dataframe(fn):
    """Return a pandas Dataframe given a Goldilocks FITS file name"""
    hdus = fits.open(fn)
    df_original = pd.DataFrame()
    header = hdus[0].header
    for j in range(28):
        df = pd.DataFrame()
        for i in range(1, 10):
            name = hdus[i].name
            df[name] = hdus[i].data[j, :]
        df["order"] = j
        df_original = df_original.append(df, ignore_index=True)
    keep_mask = df_original[df_original.columns[0:6]] != 0.0
    df_original = df_original[keep_mask.all(axis=1)].reset_index(drop=True)

    return df_original, header


def normalize_spectrum(df):
    """Normalizes spectrum to set to one"""
    for order in df.order.unique():
        mask = df.order == order
        norm_constant = df["Sci Flux"][
            mask
        ].median()  # mean takes outliers into account
        df["Sci Flux"][mask] = df["Sci Flux"][mask] / norm_constant
        df["Sci Error"][mask] = df["Sci Error"][mask] / norm_constant

    return df


def generative_model(m, b, A, mu, logw, int_wl=calcium_line, wl=None):
    """Generate the model given parameters"""
    continuum = m * (wl - int_wl) + b
    w = np.exp(logw)
    gaussian = A * np.exp(-0.5 * (wl - mu) ** 2 / w ** 2)
    return continuum - gaussian


def log_likelihood(theta, index):
    dat = preloaded_data[index]
    wl, flux, unc = dat["wl"], dat["flux"], dat["unc"]
    m, b, A, mu, logw = theta
    model = generative_model(m, b, A, mu, logw, int_wl=calcium_line, wl=wl)
    residual = flux - model
    chi_squared = np.sum(residual ** 2 / unc ** 2)
    return -0.5 * chi_squared


# Read in all the data first so it can be global
goldilocks_files = glob.glob(
    "../data/HPF/Helium-transit-data/**/Goldilocks*.fits", recursive=True
)

order = 4
n_walkers = 32
n_params = 5
n_steps = 5000
labels = ["m", "b", "A", "mu", "w"]
wavelength1 = 8538
wavelength2 = 8546

m_guess, b_guess, A_guess, mu_guess, logw_guess = (
    0.01,
    0.3,
    0.1,
    calcium_line,
    np.log(0.4),
)


df_meta = pd.DataFrame()
preloaded_data = {}
problem_files = {}

n_files = len(goldilocks_files)

for index in tqdm(range(0, n_files)):

    fn = goldilocks_files[index]

    try:
        df_orig, header = get_goldilocks_dataframe(fn)
        date_raw = header["DATE-OBS"]
        date = date_raw[0:10]
        time_val = date_raw[11:19]
        obj = header["OBJECT"]
        df = normalize_spectrum(df_orig)
        qidx = header["QIDX"]
        t = Time(date_raw, format="isot", scale="utc")
        jd = t.jd

        sub_region = (
            (df.order == order)
            & (df["Sci Wavl"] > wavelength1)
            & (df["Sci Wavl"] < wavelength2)
        )
        wl = df["Sci Wavl"][sub_region].values
        flux = df["Sci Flux"][sub_region].values
        unc = df["Sci Error"][sub_region].values

        temp_dict = {
            "date": date,
            "star_name": obj,
            "time": time_val,
            "int_wv": calcium_line,
            "qidx": qidx,
            "jd": jd,
            "fn": fn[-49:],
        }
        temp_df = pd.DataFrame(temp_dict, index=[index])
        df_meta = df_meta.append(temp_df)
        preloaded_data[index] = {"wl": wl, "flux": flux, "unc": unc}
    except:
        print("Failed to read file:", fn[-49:])


def main_function(index):

    theta_guess = np.array([m_guess, b_guess, A_guess, mu_guess, logw_guess])

    pos = theta_guess + 1e-4 * np.random.randn(n_walkers, n_params)
    sampler = emcee.EnsembleSampler(n_walkers, n_params, log_likelihood, args=(index,))
    sampler.run_mcmc(pos, n_steps, progress=False)

    flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)

    A_draws = flat_samples[:, 2]
    b_draws = flat_samples[:, 1]
    m_draws = flat_samples[:, 0]
    mu_draws = flat_samples[:, 3]
    w_draws = np.exp(flat_samples[:, 4])

    EW = (
        ((2 * np.pi) ** 0.5)
        * (A_draws * w_draws)
        / (m_draws * (mu_draws - calcium_line) + b_draws)
    )

    ew_mean = np.mean(EW)
    ew_std = np.std(EW)
    gauss_width = np.mean(w_draws)
    gauss_width_unc = np.std(w_draws)
    obs_line_center = np.mean(mu_draws)
    obs_line_center_unc = np.std(mu_draws)

    result_dict = {
        "ew": ew_mean,
        "ew_unc": ew_std,
        "gaussian_width": gauss_width,
        "gaussian_width_unc": gauss_width_unc,
        "obs_line_center": obs_line_center,
        "obs_line_center_unc": obs_line_center_unc,
    }

    return {index: result_dict}


if __name__ == "__main__":

    start = time.time()
    with Pool(12) as pool:
        results = pool.map(main_function, preloaded_data.keys())
    end = time.time()

    # Convert list of dicts to a dictionary
    dict_out = {}

    for element in results:
        for key in element:
            dict_out[key] = element[key]

    time_diff = end - start
    print("All MCMC took:  {0:.1f} seconds".format(time_diff))
    print("or:  {0:.1f} seconds/spectrum".format(time_diff / 36))

    df_results = pd.DataFrame(dict_out).T
    df_out = pd.merge(
        df_meta, df_results, left_index=True, right_index=True, how="outer"
    )

    df_out.to_csv("../data/preliminary_results_8542_Angstrom_line.csv")
