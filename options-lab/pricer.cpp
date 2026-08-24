// European Black–Scholes + Greeks, then MC / assignment on a 1x2 call ratio
// overlaid on 100 shares. MC samples S_T at expiry only.  make && ./pricer

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

struct Contract {
    double S = 100.0;
    double K = 100.0;
    double r = 0.05;
    double q = 0.00;
    double sigma = 0.20;
    double T = 1.00;
};

double norm_pdf(double x) {
    static const double inv_sqrt_2pi = 0.3989422804014327;
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

double norm_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double d1(const Contract& c) {
    return (std::log(c.S / c.K) + (c.r - c.q + 0.5 * c.sigma * c.sigma) * c.T) /
           (c.sigma * std::sqrt(c.T));
}

double d2(const Contract& c) {
    return d1(c) - c.sigma * std::sqrt(c.T);
}

struct PriceGreeks {
    double price;
    double delta;
    double gamma;
    double vega;
    double theta;
};

PriceGreeks black_scholes(const Contract& c, bool is_call) {
    const double D1 = d1(c);
    const double D2 = d2(c);
    const double disc_S = c.S * std::exp(-c.q * c.T);
    const double disc_K = c.K * std::exp(-c.r * c.T);
    const double n1 = norm_pdf(D1);
    const double sqrtT = std::sqrt(c.T);

    PriceGreeks g{};
    if (is_call) {
        g.price = disc_S * norm_cdf(D1) - disc_K * norm_cdf(D2);
        g.delta = std::exp(-c.q * c.T) * norm_cdf(D1);
        g.theta = -disc_S * n1 * c.sigma / (2.0 * sqrtT) - c.r * disc_K * norm_cdf(D2) +
                  c.q * disc_S * norm_cdf(D1);
    } else {
        g.price = disc_K * norm_cdf(-D2) - disc_S * norm_cdf(-D1);
        g.delta = -std::exp(-c.q * c.T) * norm_cdf(-D1);
        g.theta = -disc_S * n1 * c.sigma / (2.0 * sqrtT) + c.r * disc_K * norm_cdf(-D2) -
                  c.q * disc_S * norm_cdf(-D1);
    }
    g.gamma = std::exp(-c.q * c.T) * n1 / (c.S * c.sigma * sqrtT);
    g.vega = disc_S * n1 * sqrtT;
    return g;
}

PriceGreeks scale(const PriceGreeks& g, double w) {
    return {w * g.price, w * g.delta, w * g.gamma, w * g.vega, w * g.theta};
}

PriceGreeks add(const PriceGreeks& a, const PriceGreeks& b) {
    return {a.price + b.price, a.delta + b.delta, a.gamma + b.gamma, a.vega + b.vega,
            a.theta + b.theta};
}

// 1x2 call ratio on 100 shares: long 100 sh, +1 call K1, -2 calls K2.
// At expiry, physical settlement:
//   ST > K1 → exercise the long call  → +100 shares, pay 100*K1
//   ST > K2 → assigned on 2 short calls → -200 shares, receive 200*K2
// Terminal shares = 100 + 100*1_{ST>K1} - 200*1_{ST>K2}
//   ST < K1: 100 (keep stock)
//   K1 < ST < K2: 200
//   ST > K2: 0  (stock called away)
constexpr int kShares0 = 100;
constexpr double kMult = 100.0;  // one option contract = 100 shares

int terminal_shares(double ST, double K1, double K2) {
    int sh = kShares0;
    if (ST > K1) sh += 100;
    if (ST > K2) sh -= 200;
    return sh;
}

double option_payoff(double ST, double K1, double K2) {
    return std::max(ST - K1, 0.0) - 2.0 * std::max(ST - K2, 0.0);
}

struct McResult {
    double mean;
    double se;
    double p_long_ex;     // P(ST > K1)
    double p_called_away; // P(ST > K2): shorts assigned, stock gone
    double avg_shares;
};

McResult monte_carlo_ratio(const Contract& mkt, double K1, double K2, int n_paths,
                           unsigned seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> z(0.0, 1.0);
    const double drift = (mkt.r - mkt.q - 0.5 * mkt.sigma * mkt.sigma) * mkt.T;
    const double vol = mkt.sigma * std::sqrt(mkt.T);
    const double disc = std::exp(-mkt.r * mkt.T);

    double sum = 0.0, sumsq = 0.0, shares_sum = 0.0;
    int n_ex = 0, n_assign = 0;
    for (int i = 0; i < n_paths; ++i) {
        const double ST = mkt.S * std::exp(drift + vol * z(rng));
        const double pay = option_payoff(ST, K1, K2);
        sum += pay;
        sumsq += pay * pay;
        if (ST > K1) ++n_ex;
        if (ST > K2) ++n_assign;
        shares_sum += static_cast<double>(terminal_shares(ST, K1, K2));
    }
    const double n = static_cast<double>(n_paths);
    const double mean = sum / n;
    const double var = (sumsq - n * mean * mean) / (n - 1.0);
    McResult out{};
    out.mean = disc * mean;
    out.se = disc * std::sqrt(var / n);
    out.p_long_ex = n_ex / n;
    out.p_called_away = n_assign / n;
    out.avg_shares = shares_sum / n;
    return out;
}

double monte_carlo_call(const Contract& c, int n_paths, unsigned seed, double& se) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> z(0.0, 1.0);
    const double drift = (c.r - c.q - 0.5 * c.sigma * c.sigma) * c.T;
    const double vol = c.sigma * std::sqrt(c.T);
    const double disc = std::exp(-c.r * c.T);
    double sum = 0.0, sumsq = 0.0;
    for (int i = 0; i < n_paths; ++i) {
        const double ST = c.S * std::exp(drift + vol * z(rng));
        const double pay = std::max(ST - c.K, 0.0);
        sum += pay;
        sumsq += pay * pay;
    }
    const double n = static_cast<double>(n_paths);
    const double mean = sum / n;
    const double var = (sumsq - n * mean * mean) / (n - 1.0);
    se = disc * std::sqrt(var / n);
    return disc * mean;
}

int main() {
    const Contract atm{};
    const auto call = black_scholes(atm, true);
    const auto put = black_scholes(atm, false);
    const double parity = std::abs((call.price - put.price) -
                                   (atm.S * std::exp(-atm.q * atm.T) - atm.K * std::exp(-atm.r * atm.T)));

    constexpr int N = 1000000;
    double se_call = 0.0;
    const double mc_call = monte_carlo_call(atm, N, 42, se_call);
    const double err_call = std::abs(mc_call - call.price);

    // 1x2: +1 ATM call, -2 calls 5% OTM, on 100 shares
    Contract k2 = atm;
    k2.K = atm.S * 1.05;
    const auto long_c = black_scholes(atm, true);
    const auto short_c = black_scholes(k2, true);
    const auto ratio = add(scale(long_c, 1.0), scale(short_c, -2.0));
    // stock overlay: +100 shares → extra delta +1 (per share), value 100*S
    const double pkg_delta = kShares0 / kMult + ratio.delta;  // per-share delta of whole pkg

    const McResult mc = monte_carlo_ratio(atm, atm.K, k2.K, N, 42);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== Vanilla BS check ===\n";
    std::cout << "S=K=" << atm.S << " r=" << atm.r << " sigma=" << atm.sigma << " T=" << atm.T
              << "\nCall " << call.price << "  Put " << put.price << "\n";
    std::cout << "Delta " << call.delta << "  Gamma " << call.gamma << "  Vega " << call.vega
              << "  Theta " << call.theta << "\n";
    std::cout << "Parity err " << parity << "\n";
    std::cout << "MC vanilla call N=" << N << "  " << mc_call << "  |BS-MC|=" << err_call
              << (err_call < 0.05 ? "  PASS\n\n" : "  FAIL\n\n");

    std::cout << "=== 1x2 call ratio on 100 shares ===\n";
    std::cout << "+100 sh  +1 call K=" << atm.K << "  -2 calls K=" << k2.K << "\n";
    std::cout << "Option package  value " << ratio.price << "  delta " << ratio.delta
              << "  vega " << ratio.vega << "  theta " << ratio.theta << "\n";
    std::cout << "With stock       delta " << pkg_delta << " per share-equivalent\n";
    std::cout << "MC option pkg    " << mc.mean << "  se " << mc.se
              << "  |BS-MC|=" << std::abs(mc.mean - ratio.price) << "\n";
    std::cout << "P(long call exercised)     " << mc.p_long_ex << "\n";
    std::cout << "P(shorts assigned, stock called away) " << mc.p_called_away << "\n";
    std::cout << "E[shares at expiry]        " << mc.avg_shares << "   (start 100; 0 if ST>K2)\n\n";

    std::cout << "BS ratio value surface (option pkg only), rows=S  cols=vol 15/20/25%\n";
    std::cout << std::setprecision(3);
    const double spots[] = {90.0, 95.0, 100.0, 105.0, 110.0};
    const double vols[] = {0.15, 0.20, 0.25};
    std::cout << "    S\\vol   15%     20%     25%\n";
    for (double S : spots) {
        std::cout << "    " << std::setw(5) << S;
        for (double sig : vols) {
            Contract a = atm;
            a.S = S;
            a.sigma = sig;
            Contract b = a;
            b.K = 105.0;
            a.K = 100.0;
            const double v = black_scholes(a, true).price - 2.0 * black_scholes(b, true).price;
            std::cout << "  " << std::setw(7) << v;
        }
        std::cout << "\n";
    }

    return err_call < 0.05 ? 0 : 1;
}
