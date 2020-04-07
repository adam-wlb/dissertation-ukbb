import java.io.*;
import java.util.Random;

/**
 * @author Adam Barron
 * @author 160212899
 * Generate a subset of the biobank data according to some probability.
 */
public class GenerateSubset {

    public static void main(String[] args) {
        if (args.length != 3)
            throw new IllegalArgumentException("Usage: <class_labels> <probability> <output_file>");
        String classLabels = args[0];
        double prob = Double.parseDouble(args[1]);
        if (prob >= 1 || prob <= 0)
            throw new IllegalArgumentException("Probability must be between 0 and 1");
        String outFile = args[2];

        try {
            BufferedReader br = new BufferedReader(new FileReader(classLabels));
            BufferedWriter bw = new BufferedWriter(new FileWriter(outFile));
            // Copy header line to output
            bw.write(br.readLine() + "\n");
            String currentLine;
            int count = 0;
            while ((currentLine = br.readLine()) != null) {
                double random = new Random().nextDouble();
                if (random < prob) {
                    bw.write(currentLine + "\n");
                    count++;
                }
            }
            bw.close();
            br.close();
            System.out.println("Created subset of size " + count);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
