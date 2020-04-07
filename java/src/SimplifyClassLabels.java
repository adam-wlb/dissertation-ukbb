import java.io.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @author Adam Barron
 * @author 160212899
 * Simplifies class labels. All unhealthy patients (1,2,3) become class 1, all healthy patients (0) remain the same.
 */
public class SimplifyClassLabels {

    public static void main(String[] args) {
        if (args.length != 1)
            throw new IllegalArgumentException("Usage: <class_labels>");
        String input = args[0];
        // Look for unhealthy patients
        Pattern pat = Pattern.compile(",[123]");
        Matcher mat = pat.matcher("");

        try {
            BufferedReader br = new BufferedReader(new FileReader(input));
            BufferedWriter bw = new BufferedWriter(new FileWriter("simplified-" + input));
            // Copy header line to output
            bw.write(br.readLine() + "\n");
            String currentLine;
            while ((currentLine = br.readLine()) != null) {
                mat.reset(currentLine);
                // If class 1, 2 or 3
                if (mat.find()) {
                    String lineSplit[] = currentLine.split(",");
                    String id = lineSplit[0];
                    // Change class to 1
                    bw.write(id + ",1" + "\n");
                } else {
                    bw.write(currentLine + "\n");
                }
            }
            bw.close();
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
