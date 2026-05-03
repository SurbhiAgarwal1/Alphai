def calculate_metrics(results, alpha=0.05):
    """
    Calculate Coverage, Average Width, and Winkler Score.
    results: list of dicts with 'lower', 'upper', 'actual'
    """
    coverage_count = 0
    total_width = 0
    total_winkler = 0
    
    for res in results:
        L = res['lower']
        U = res['upper']
        Y = res['actual']
        
        width = U - L
        total_width += width
        
        if L <= Y <= U:
            coverage_count += 1
            winkler = width
        elif Y < L:
            winkler = width + (2 / alpha) * (L - Y)
        else:
            winkler = width + (2 / alpha) * (Y - U)
            
        total_winkler += winkler
        
    n = len(results)
    if n == 0:
        return 0.0, 0.0, 0.0
        
    coverage = coverage_count / n
    avg_width = total_width / n
    avg_winkler = total_winkler / n
    
    return coverage, avg_width, avg_winkler
