import java.io.*;

/**
 * @author Adam Barron
 * @author 160212899
 * Produce a CSV file of IDs for all patients with accelerometer data.
 * Can take 2 args to run over data1 and data2 into one csv.
 */
public class GetAccelerometerIDs {

    private static BufferedWriter bw;

    public static void main(String[] args) {
        if (args.length < 2)
            throw new IllegalArgumentException("Usage: <output_file> <filepath_1> <optional_filepath>");
        String outputFile = args[0];
        File[] files;
        try {
            bw = new BufferedWriter(new FileWriter(outputFile));
        } catch (IOException e) {
            e.printStackTrace();
        }

        String filePath = args[1];
        files = new File(filePath).listFiles();
        try {
            appendIDs(files);
        } catch (IOException e) {
            e.printStackTrace();
        }

        if (args[2] != null) {
            filePath = args[2];
            files = new File(filePath).listFiles();
            try {
                appendIDs(files);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        try {
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void appendIDs(File[] files) throws IOException {
        for (File f : files) {
            if (f.isDirectory()) {
                // Recurse over folders
                appendIDs(f.listFiles());
            } else {
                // Split on "_" to get ID component of file name
                String[] filename = f.getName().split("_");
                String id = filename[0];
                bw.write(id + "\n");
            }
        }
    }
}
