import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Points, PointMaterial } from '@react-three/drei';
import * as THREE from 'three';

interface NeuralBackgroundProps {
  speedUp?: boolean;
}

function ParticleField({ speedUp }: { speedUp: boolean }) {
  const ref = useRef<THREE.Points>(null);
  
  // Generate random particles in a sphere
  const sphere = useMemo(() => {
    const count = 3000;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const r = 10 * Math.cbrt(Math.random());
      const theta = Math.random() * 2 * Math.PI;
      const phi = Math.acos(2 * Math.random() - 1);
      
      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = r * Math.cos(phi);
    }
    return positions;
  }, []);

  useFrame((_, delta) => {
    if (ref.current) {
      // Base rotation speed
      let rotationSpeed = delta / 15;
      
      // Speed up if hovered
      if (speedUp) {
        rotationSpeed *= 8;
      }
      
      ref.current.rotation.x -= rotationSpeed;
      ref.current.rotation.y -= rotationSpeed;
    }
  });

  return (
    <group rotation={[0, 0, Math.PI / 4]}>
      <Points ref={ref} positions={sphere} stride={3} frustumCulled={false}>
        <PointMaterial
          transparent
          color="#00f0ff"
          size={0.03}
          sizeAttenuation={true}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </Points>
      <Points positions={sphere} stride={3} frustumCulled={false} rotation={[Math.PI, 0, 0]}>
        <PointMaterial
          transparent
          color="#00ff66"
          size={0.02}
          sizeAttenuation={true}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
          opacity={0.6}
        />
      </Points>
    </group>
  );
}

export function NeuralBackground({ speedUp = false }: NeuralBackgroundProps) {
  return (
    <div className="fixed inset-0 z-[-40] pointer-events-none">
      <Canvas camera={{ position: [0, 0, 8] }} gl={{ alpha: true }}>
        <ParticleField speedUp={speedUp} />
      </Canvas>
    </div>
  );
}
