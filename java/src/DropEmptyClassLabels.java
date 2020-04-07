import java.io.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @author Adam Barron
 * @author 160212899
 * Drop any patients with no class label.
 */
public class DropEmptyClassLabels {

    public static void main(String[] args) {
        if (args.length != 2)
            throw new IllegalArgumentException("Usage: <input_file> <output_file>");
        String inputFile = args[0];
        String outputFile = args[1];

        BufferedReader br = null;
        BufferedWriter bw = null;
        // Look for patients with no class labels
        Pattern pat = Pattern.compile(", ");
        Matcher mat = pat.matcher("");
        try {
            br = new BufferedReader(new FileReader(inputFile));
            bw = new BufferedWriter(new FileWriter(outputFile));
            String currentLine;
            while ((currentLine = br.readLine()) != null) {
                mat.reset(currentLine);
                if (!(mat.find())) {
                    bw.write(currentLine + "\n");
                }
            }
            bw.close();
            br.close();
        } catch (FileNotFoundException e) {
            System.out.println("File " + inputFile + " not found");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
