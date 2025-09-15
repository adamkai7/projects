#include <iostream>
#include <cmath>
// Adam Kainikara  
// Contact: adamkainikara@gmail.com

// The purpose of this code is to calculate the call and put option prices using the Black Scholes model in C++
// This is the first major program I've done in C++, it is a language that Im very much am still a beginner at
// However there are steps being made for me to become more advanced in this language 


// How to run: g++ black_scholes.cpp -o black_scholes -std=c++11 -lm
// ./black_scholes
///Underlying Price (S): 100
//Strike Price (K): 100
//Risk free rate (r): 0.05
//Implied Volatility (sigma): 0.2
//Time to Expiry (in days): 30
//Call Option Price: 2.49338
//Put Option Price: 2.08326





// cumulative distribution function for normal distribution
double norm_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

// Black Scholes formula for call option
double black_scholes_call(double S, double K, double r, double sigma, double T) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

// Black Scholes formula for put option
double black_scholes_put(double S, double K, double r, double sigma, double T) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
}

int main() {
    double S, K, r, sigma;
    int days;

    std::cout << "Underlying Price (S): ";
    std::cin >> S;
    std::cout << "Strike Price (K): ";
    std::cin >> K;
    std::cout << "Risk free rate (r): ";
    std::cin >> r;
    std::cout << "Implied Volatility (sigma): ";
    std::cin >> sigma;
    std::cout << "Time to Expiry (in days): ";
    std::cin >> days;

    // converts days to years (assuming 365 days per year)
    double T = days / 365.0;

    double call_price = black_scholes_call(S, K, r, sigma, T);
    double put_price  = black_scholes_put(S, K, r, sigma, T);

    std::cout << "Call Option Price: " << call_price << std::endl;
    std::cout << "Put Option Price: " << put_price << std::endl;

    return 0;
}
