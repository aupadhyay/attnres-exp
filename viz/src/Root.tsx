import { Composition } from "remotion";
import { CodeRouting } from "./CodeRouting";

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="CodeRouting"
      component={CodeRouting}
      durationInFrames={240}
      fps={30}
      width={1000}
      height={640}
    />
  );
};
