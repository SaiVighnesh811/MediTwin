import Nav from '../components/layout/Nav';
import Footer from '../components/layout/Footer';
import Hero from '../sections/Hero';
import Pipeline from '../sections/Pipeline';
import IODiagram from '../sections/IODiagram';
import HonestMetrics from '../sections/HonestMetrics';
import Responsible from '../sections/Responsible';
import FinalCTA from '../sections/FinalCTA';

export default function Home() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <Pipeline />
        <IODiagram />
        <HonestMetrics />
        <Responsible />
        <FinalCTA />
      </main>
      <Footer />
    </>
  );
}

