import SimpleITK as sitk


def intensity_harmonization(img1: sitk.Image, img2: sitk.Image):
    # Create a statistics filter
    stats_filter = sitk.StatisticsImageFilter()
    # Compute statistics (including standard deviation)
    stats_filter.Execute(img1)
    # Get standard deviation
    sigma = stats_filter.GetSigma()
    mean = stats_filter.GetMean()