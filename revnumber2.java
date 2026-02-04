public class revnumber2{
    public static void main(String args[]){
        int n = 861993;
        int rev = 0;
        while(n>0){
            int lastdigit = n%10;
            rev = (rev* 10)+ lastdigit;
            n = n/10;
        }
        System.out.print(rev);
    }
}
