extern crate num_complex;

use num_complex::Complex;

const EPSILON: f64 = 1e-10;

fn differential<F>(f: F) -> impl Fn(f64) -> f64
where
    F: Fn(Complex<f64>) -> Complex<f64>,
{
    move |x| {
        f(Complex::new(x, EPSILON)).im / EPSILON
    }
}


fn main() {
    let dx = differential(|x| x.exp() / (x.sin().powf(3.) + x.cos().powf(3.)).sqrt());
    println!("{}", dx(1.5));
}
